module DCDataset

using CSV, DataFrames, Ipopt, JuMP, LinearAlgebra, MathOptInterface
using PowerModels, ProgressMeter, Random, SparseArrays, Statistics

const MOI = MathOptInterface

function dc_parameters(case_path::String)
    data = parse_file(case_path)
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

    gen_ids = sort(collect(keys(ref[:gen])))
    branch_ids = sort(collect(keys(ref[:branch])))
    bus_ids = sort(collect(keys(ref[:bus])))
    bus_lookup = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))
    bus_count = length(bus_ids)

    p_min = [ref[:gen][i]["pmin"] for i in gen_ids]
    p_max = [ref[:gen][i]["pmax"] for i in gen_ids]
    f_max = [get(ref[:branch][i], "rate_a", Inf) for i in branch_ids]

    cost_c2, cost_c1, cost_c0 = Float64[], Float64[], Float64[]
    for gen_id in gen_ids
        cost_coeffs = ref[:gen][gen_id]["cost"]
        if length(cost_coeffs) == 3
            push!(cost_c2, cost_coeffs[1])
            push!(cost_c1, cost_coeffs[2])
            push!(cost_c0, cost_coeffs[3])
        elseif length(cost_coeffs) == 2
            push!(cost_c2, 0.0)
            push!(cost_c1, cost_coeffs[1])
            push!(cost_c0, cost_coeffs[2])
        else
            push!(cost_c2, 0.0)
            push!(cost_c1, 0.0)
            push!(cost_c0, 0.0)
        end
    end

    Bbus = zeros(bus_count, bus_count)
    for (i, branch) in ref[:branch]
        f_bus, t_bus = bus_lookup[branch["f_bus"]], bus_lookup[branch["t_bus"]]
        b = -1 / branch["br_x"]
        Bbus[f_bus, t_bus] += b
        Bbus[t_bus, f_bus] += b
        Bbus[f_bus, f_bus] -= b
        Bbus[t_bus, t_bus] -= b
    end
    for (i, bus) in ref[:bus]
        Bbus[bus_lookup[i], bus_lookup[i]] += get(bus, "bs", 0.0)
    end

    branch_count = length(branch_ids)
    A = spzeros(Int, branch_count, bus_count)
    b_diag = spzeros(Float64, branch_count, branch_count)
    for (i, br_id) in enumerate(branch_ids)
        branch = ref[:branch][br_id]
        f_bus, t_bus = bus_lookup[branch["f_bus"]], bus_lookup[branch["t_bus"]]
        A[i, f_bus] = 1
        A[i, t_bus] = -1
        b_diag[i, i] = -1 / branch["br_x"]
    end

    slack_bus_idx = bus_lookup[first(collect(keys(ref[:ref_buses])))]
    non_slack_indices = [i for i in 1:bus_count if i != slack_bus_idx]
    Bbus_ns = Bbus[non_slack_indices, non_slack_indices]
    A_ns = A[:, non_slack_indices]
    ptdf_matrix_ns = b_diag * A_ns * inv(Matrix(Bbus_ns))
    ptdf_matrix = zeros(branch_count, bus_count)
    ptdf_matrix[:, non_slack_indices] = ptdf_matrix_ns

    gen_count = length(gen_ids)
    bus_gen_map = zeros(Int, bus_count, gen_count)
    for (i, gen_id) in enumerate(gen_ids)
        gen_bus = ref[:gen][gen_id]["gen_bus"]
        bus_pos = bus_lookup[gen_bus]
        bus_gen_map[bus_pos, i] = 1
    end

    return ( p_min=p_min, p_max=p_max, f_max=f_max, cost_c1=cost_c1, cost_c2=cost_c2, cost_c0=cost_c0,
             ptdf=ptdf_matrix, bus_gen_map=bus_gen_map, gen_ids=gen_ids, bus_ids=bus_ids,
             branch_ids=branch_ids, bus_lookup=bus_lookup )
end

function generate_dc_dataset(case_path::String, output_dir::String;
                             num_samples::Int=50000, variance::Float64=0.12, seed::Int=42,
                             save_intermediate_files::Bool=true)
    num_samples > 0 || error("num_samples must be positive")
    isfinite(variance) && variance >= 0 || error("variance must be finite and nonnegative")
    ispath(output_dir) && error("Output path already exists: $output_dir")
    Random.seed!(seed)
    println("Generating dataset for $(basename(case_path))")
    println("Attempts: $num_samples, Gaussian spread: $variance, seed: $seed")

    params = dc_parameters(case_path)

    data = parse_file(case_path)
    base_loads = Dict(load["load_bus"] => load["pd"] for (load_idx, load) in data["load"])
    load_bus_indices = sort(collect(keys(base_loads)))

    valid_branch_indices = findall(params.f_max .< 1e10)
    ptdf_constrained = params.ptdf[valid_branch_indices, :]
    f_max_constrained = params.f_max[valid_branch_indices]

    num_gens = length(params.gen_ids)
    num_buses = length(params.bus_ids)
    num_branches_constrained = length(f_max_constrained)

    model = Model(Ipopt.Optimizer)
    set_silent(model)

    @variable(model, pg[1:num_gens])
    @variable(model, pd[1:num_buses])
    @objective(model, Min, sum(params.cost_c2[i]*pg[i]^2 + params.cost_c1[i]*pg[i] + params.cost_c0[i] for i in 1:num_gens))
    @constraint(model, power_balance, sum(pg) == sum(pd))
    @constraint(model, gen_min[i=1:num_gens], pg[i] >= params.p_min[i])
    @constraint(model, gen_max[i=1:num_gens], pg[i] <= params.p_max[i])
    @expression(model, pinj[j=1:num_buses], sum(params.bus_gen_map[j, k] * pg[k] for k in 1:num_gens) - pd[j])
    @constraint(model, line_pos[i=1:num_branches_constrained], sum(ptdf_constrained[i, j] * pinj[j] for j in 1:num_buses) <= f_max_constrained[i])
    @constraint(model, line_neg[i=1:num_branches_constrained], sum(ptdf_constrained[i, j] * pinj[j] for j in 1:num_buses) >= -f_max_constrained[i])

    load_data_list, generation_data_list = Vector{Float64}[], Vector{Float64}[]
    lambda_list, cost_list = Float64[], Float64[]
    mu_g_min_list, mu_g_max_list = Vector{Float64}[], Vector{Float64}[]
    mu_line_pos_list, mu_line_neg_list = Vector{Float64}[], Vector{Float64}[]

    total_loop_time = @elapsed begin
        @showprogress "Generating Samples" for _ in 1:num_samples
            new_loads = Dict{Int, Float64}()
            for bus_idx in load_bus_indices
                base_pd = base_loads[bus_idx]
                sigma = abs(base_pd * variance)
                new_loads[bus_idx] = max(0.0, base_pd + sigma * randn())
            end

            current_loads = zeros(num_buses)
            for (bus_id, load_val) in new_loads
                bus_pos = params.bus_lookup[bus_id]
                current_loads[bus_pos] = load_val
            end
            for j in 1:num_buses
                fix(pd[j], current_loads[j])
            end

            optimize!(model)
            if termination_status(model) in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
                push!(cost_list, objective_value(model))
                push!(load_data_list, [new_loads[i] for i in load_bus_indices])
                push!(generation_data_list, value.(pg))
                push!(lambda_list, dual(power_balance))
                push!(mu_g_min_list, dual.(gen_min))
                push!(mu_g_max_list, -dual.(gen_max))
                push!(mu_line_pos_list, -dual.(line_pos))
                push!(mu_line_neg_list, dual.(line_neg))
            end
        end
    end
    successful_samples = length(cost_list)
    println("Successfully generated $(successful_samples) / $(num_samples) samples in $(round(total_loop_time, digits=2))s.")
    if successful_samples > 0
        println("Average cost: ", round(mean(cost_list), digits=4), "\$")
        println("Average solve time: ", round((total_loop_time / successful_samples) * 1000, digits=2), "ms")
    end

    successful_samples > 0 || error("No solved DC samples; no files written")
    mkpath(output_dir)
    case_name = splitext(basename(case_path))[1]

    df_loads = DataFrame(hcat(load_data_list...)', Symbol.("pd" .* string.(load_bus_indices)))
    df_gens = DataFrame(hcat(generation_data_list...)', Symbol.("pg" .* string.(params.gen_ids)))
    df_lambda = DataFrame(lambda = lambda_list)
    df_mu_g_min = DataFrame(hcat(mu_g_min_list...)', Symbol.("mu_g_min_" .* string.(params.gen_ids)))
    df_mu_g_max = DataFrame(hcat(mu_g_max_list...)', Symbol.("mu_g_max_" .* string.(params.gen_ids)))

    df_mu_line_pos = DataFrame(hcat(mu_line_pos_list...)', Symbol.("mu_line_max_" .* string.(params.branch_ids[valid_branch_indices])))
    df_mu_line_neg = DataFrame(hcat(mu_line_neg_list...)', Symbol.("mu_line_min_" .* string.(params.branch_ids[valid_branch_indices])))

    if save_intermediate_files
        CSV.write(joinpath(output_dir, "$(case_name)_loads.csv"), df_loads)
        CSV.write(joinpath(output_dir, "$(case_name)_generations.csv"), df_gens)
    end

    final_df = hcat(df_loads, df_gens, df_lambda, df_mu_g_min, df_mu_g_max, df_mu_line_pos, df_mu_line_neg)
    final_dataset_path = joinpath(output_dir, "$(case_name)_dataset_with_duals.csv")
    CSV.write(final_dataset_path, final_df)

    println("Dataset saved to: $(final_dataset_path)")
    return successful_samples
end

function main(args=ARGS)
    usage = "Usage: julia --project=data_generalization data_generalization/generate_dc_dataset.jl CASE_FILE OUTPUT_DIR [SAMPLES=50000] [VARIANCE=0.12] [SEED=42]"
    if args == ["--help"]
        println(usage)
        return
    end
    2 <= length(args) <= 5 || error(usage)
    generate_dc_dataset(args[1], args[2];
        num_samples=length(args) >= 3 ? parse(Int, args[3]) : 50000,
        variance=length(args) >= 4 ? parse(Float64, args[4]) : 0.12,
        seed=length(args) >= 5 ? parse(Int, args[5]) : 42)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    DCDataset.main()
end
