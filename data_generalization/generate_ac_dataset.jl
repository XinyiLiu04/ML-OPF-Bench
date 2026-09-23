module ACDataset

using CSV, DataFrames, Ipopt, MathOptInterface, PowerModels, ProgressMeter, Random, Statistics

const MOI = MathOptInterface

function generate_ac_dataset(case_path::String, output_dir::String;
                             num_samples::Int=50000, variance::Float64=0.12, seed::Int=42)
    num_samples > 0 || error("num_samples must be positive")
    isfinite(variance) && variance >= 0 || error("variance must be finite and nonnegative")
    ispath(output_dir) && error("Output path already exists: $output_dir")
    Random.seed!(seed)
    data = PowerModels.parse_file(case_path)
    base_pd = Dict(load["load_bus"] => load["pd"] for load in values(data["load"]) if load["pd"] > 0)
    base_qd = Dict(load["load_bus"] => load["qd"] for load in values(data["load"]) if haskey(load, "qd"))
    load_ids = sort(collect(keys(base_pd)))
    gen_ids = sort([g["index"] for g in values(data["gen"])])
    bus_ids = sort([b["bus_i"] for b in values(data["bus"])])
    optimizer = MOI.OptimizerWithAttributes(Ipopt.Optimizer, "print_level" => 0, "sb" => "yes")
    rows = (pd=Vector{Float64}[], qd=Vector{Float64}[], pg=Vector{Float64}[],
            qg=Vector{Float64}[], vm=Vector{Float64}[], va=Vector{Float64}[])
    costs = Float64[]
    println("Attempts: $num_samples, Gaussian spread: $variance, seed: $seed")

    elapsed = @elapsed begin
        @showprogress "Generating AC samples" for _ in 1:num_samples
            sample = deepcopy(data)
            pd = Dict{Int, Float64}()
            qd = Dict{Int, Float64}()
            for id in load_ids
                pd[id] = max(0.0, base_pd[id] + abs(base_pd[id] * variance) * randn())
                q = get(base_qd, id, 0.0)
                qd[id] = q + abs(q * variance) * randn()
            end
            for load in values(sample["load"])
                id = load["load_bus"]
                if haskey(pd, id)
                    load["pd"] = pd[id]
                    load["qd"] = qd[id]
                end
            end
            result = solve_ac_opf(sample, optimizer;
                                  setting=Dict("output" => Dict("branch_flows" => true)))
            result["termination_status"] == MOI.LOCALLY_SOLVED || continue
            solution = result["solution"]
            push!(costs, result["objective"])
            push!(rows.pd, [pd[id] for id in load_ids])
            push!(rows.qd, [qd[id] for id in load_ids])
            push!(rows.pg, [solution["gen"][string(id)]["pg"] for id in gen_ids])
            push!(rows.qg, [solution["gen"][string(id)]["qg"] for id in gen_ids])
            push!(rows.vm, [solution["bus"][string(id)]["vm"] for id in bus_ids])
            push!(rows.va, [solution["bus"][string(id)]["va"] for id in bus_ids])
        end
    end
    isempty(costs) && error("No locally solved AC samples; no files written")
    case_name = splitext(basename(case_path))[1]
    mkpath(output_dir)
    for (suffix, ids, prefix) in ((:pd, load_ids, "pd"), (:qd, load_ids, "qd"),
                                 (:pg, gen_ids, "pg_"), (:qg, gen_ids, "qg_"),
                                 (:vm, bus_ids, "vm_"), (:va, bus_ids, "va_"))
        table = DataFrame(hcat(getproperty(rows, suffix)...)', Symbol.(prefix .* string.(ids)))
        CSV.write(joinpath(output_dir, "$(case_name)_$(suffix).csv"), table)
    end
    println("Saved $(length(costs)) / $num_samples samples to $output_dir")
    println("Elapsed: $(round(elapsed; digits=2)) s; mean objective: $(mean(costs))")
    return length(costs)
end

function main(args=ARGS)
    usage = "Usage: julia --project=data_generalization data_generalization/generate_ac_dataset.jl CASE_FILE OUTPUT_DIR [SAMPLES=50000] [VARIANCE=0.12] [SEED=42]"
    if args == ["--help"]
        println(usage)
        return
    end
    2 <= length(args) <= 5 || error(usage)
    generate_ac_dataset(args[1], args[2];
        num_samples=length(args) >= 3 ? parse(Int, args[3]) : 50000,
        variance=length(args) >= 4 ? parse(Float64, args[4]) : 0.12,
        seed=length(args) >= 5 ? parse(Int, args[5]) : 42)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    ACDataset.main()
end
