module ACDuals

using CSV, DataFrames, Ipopt, JuMP, MathOptInterface, PowerModels, ProgressMeter

const MOI = MathOptInterface

function bound_duals(variables, ids, bound)
    if bound == :lower
        return [has_lower_bound(variables[id]) ? dual(LowerBoundRef(variables[id])) : 0.0 for id in ids]
    end
    return [has_upper_bound(variables[id]) ? dual(UpperBoundRef(variables[id])) : 0.0 for id in ids]
end

function generate_ac_duals(case_path::String, dataset_dir::String, output_dir::String)
    ispath(output_dir) && error("Output path already exists: $output_dir")
    case_name = splitext(basename(case_path))[1]
    pd = CSV.read(joinpath(dataset_dir, "$(case_name)_pd.csv"), DataFrame)
    qd = CSV.read(joinpath(dataset_dir, "$(case_name)_qd.csv"), DataFrame)
    nrow(pd) > 0 || error("Input dataset is empty")
    size(pd) == size(qd) || error("Pd and Qd shapes differ")
    load_ids = sort([parse(Int, name[3:end]) for name in names(pd)])
    names(pd) == "pd" .* string.(load_ids) || error("Pd columns must be sorted by bus ID")
    names(qd) == "qd" .* string.(load_ids) || error("Qd columns must match Pd bus order")

    data = PowerModels.parse_file(case_path)
    gen_ids = sort([g["index"] for g in values(data["gen"])])
    bus_ids = sort([b["bus_i"] for b in values(data["bus"])])
    branch_ids = sort([br["index"] for br in values(data["branch"])])
    limited_ids = [id for id in branch_ids if
                   0 < get(data["branch"][string(id)], "rate_a", 0.0) < Inf]
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "sb" => "yes")
    fields = (:mu_pg_min, :mu_pg_max, :mu_qg_min, :mu_qg_max, :mu_vm_min,
              :mu_vm_max, :lambda_kcl_r, :lambda_kcl_i, :mu_sm_fr, :mu_sm_to)
    rows = Dict(field => Vector{Float64}[] for field in fields)

    @showprogress "Extracting AC duals" for row in 1:nrow(pd)
        sample = deepcopy(data)
        pd_values = Dict(id => pd[row, col] for (col, id) in enumerate(load_ids))
        qd_values = Dict(id => qd[row, col] for (col, id) in enumerate(load_ids))
        for load in values(sample["load"])
            id = load["load_bus"]
            if haskey(pd_values, id)
                load["pd"] = pd_values[id]
                load["qd"] = qd_values[id]
            end
        end
        pm = instantiate_model(sample, ACPPowerModel, PowerModels.build_opf;
                               setting=Dict("output" => Dict("duals" => true)))
        result = optimize_model!(pm; optimizer=optimizer)
        result["termination_status"] == MOI.LOCALLY_SOLVED ||
            error("AC dual solve failed at input row $row: $(result["termination_status"])")
        variables = pm.var[:it][:pm][:nw][0]
        solution = result["solution"]
        for (field, variable, ids, bound) in (
            (:mu_pg_min, :pg, gen_ids, :lower), (:mu_pg_max, :pg, gen_ids, :upper),
            (:mu_qg_min, :qg, gen_ids, :lower), (:mu_qg_max, :qg, gen_ids, :upper),
            (:mu_vm_min, :vm, bus_ids, :lower), (:mu_vm_max, :vm, bus_ids, :upper))
            push!(rows[field], bound_duals(variables[variable], ids, bound))
        end
        push!(rows[:lambda_kcl_r], [solution["bus"][string(id)]["lam_kcl_r"] for id in bus_ids])
        push!(rows[:lambda_kcl_i], [solution["bus"][string(id)]["lam_kcl_i"] for id in bus_ids])

        constraints = all_constraints(pm.model, QuadExpr, MOI.LessThan{Float64})
        length(constraints) == 2 * length(limited_ids) || error("Unexpected thermal constraint count")
        mu_fr = zeros(length(branch_ids))
        mu_to = zeros(length(branch_ids))
        positions = Dict(id => i for (i, id) in enumerate(branch_ids))
        for (i, id) in enumerate(limited_ids)
            mu_fr[positions[id]] = dual(constraints[2i - 1])
            mu_to[positions[id]] = dual(constraints[2i])
        end
        push!(rows[:mu_sm_fr], mu_fr)
        push!(rows[:mu_sm_to], mu_to)
    end

    mkpath(output_dir)
    for field in fields
        ids = field in (:mu_pg_min, :mu_pg_max, :mu_qg_min, :mu_qg_max) ? gen_ids :
              field in (:mu_sm_fr, :mu_sm_to) ? branch_ids : bus_ids
        table = DataFrame(hcat(rows[field]...)', Symbol.("$(field)_" .* string.(ids)))
        CSV.write(joinpath(output_dir, "$(case_name)_$(field).csv"), table)
    end
    println("Saved duals for $(nrow(pd)) rows to $output_dir")
    return nrow(pd)
end

function main(args=ARGS)
    usage = "Usage: julia --project=data_generalization data_generalization/generate_ac_duals.jl CASE_FILE DATASET_DIR OUTPUT_DIR"
    if args == ["--help"]
        println(usage)
        return
    end
    length(args) == 3 || error(usage)
    generate_ac_duals(args...)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    ACDuals.main()
end
