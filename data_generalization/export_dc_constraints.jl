module DCConstraints

using PowerModels, DataFrames, CSV, LinearAlgebra, SparseArrays

function export_dc_constraints(case_path::String, output_dir::String)

    ispath(output_dir) && error("Output path already exists: $output_dir")
    mkpath(output_dir)
    case_name = splitext(basename(case_path))[1]

    data = parse_file(case_path)
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

    gen = sort(collect(keys(ref[:gen])))
    gen_pmin = [ref[:gen][i]["pmin"] for i in gen]
    gen_pmax = [ref[:gen][i]["pmax"] for i in gen]
    df_gen_limits = DataFrame(gen_id=gen, pgmin=gen_pmin, pgmax=gen_pmax)
    CSV.write(joinpath(output_dir, "$(case_name)_gen_limits.csv"), df_gen_limits)

    branch = sort(collect(keys(ref[:branch])))
    branch_rate_a = [get(ref[:branch][i], "rate_a", Inf) for i in branch]

    branch_f_bus = [ref[:branch][i]["f_bus"] for i in branch]
    branch_t_bus = [ref[:branch][i]["t_bus"] for i in branch]
    branch_r = [ref[:branch][i]["br_r"] for i in branch]
    branch_x = [ref[:branch][i]["br_x"] for i in branch]

    df_branch_info = DataFrame(
        branch_id=branch,
        f_bus=branch_f_bus,
        t_bus=branch_t_bus,
        r_pu=branch_r,
        x_pu=branch_x,
        rate_a=branch_rate_a
    )
    CSV.write(joinpath(output_dir, "$(case_name)_branch_info.csv"), df_branch_info)

    df_branch_limits = DataFrame(branch_id=branch, rate_a=branch_rate_a)
    CSV.write(joinpath(output_dir, "$(case_name)_branch_limits.csv"), df_branch_limits)

    bus = sort(collect(keys(ref[:bus])))
    bus_lookup = Dict(bus_id => i for (i, bus_id) in enumerate(bus))
    bus_count = length(bus)

    cost_c2, cost_c1, cost_c0 = Float64[], Float64[], Float64[]
    for gen_id in gen
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
    df_gen_costs = DataFrame(gen_id=gen, cost_c2=cost_c2, cost_c1=cost_c1, cost_c0=cost_c0)
    CSV.write(joinpath(output_dir, "$(case_name)_gen_costs.csv"), df_gen_costs)

    Bbus = zeros(bus_count, bus_count)
    for (i, branch) in ref[:branch]
        f_bus = bus_lookup[branch["f_bus"]]
        t_bus = bus_lookup[branch["t_bus"]]
        b = -1 / branch["br_x"]
        Bbus[f_bus, t_bus] += b
        Bbus[t_bus, f_bus] += b
        Bbus[f_bus, f_bus] -= b
        Bbus[t_bus, t_bus] -= b
    end

    for (i, bus) in ref[:bus]
        bus_idx = bus_lookup[i]
        Bbus[bus_idx, bus_idx] += get(bus, "bs", 0.0)
    end

    branch_count = length(branch)
    A = spzeros(Int, branch_count, bus_count)
    b_diag = spzeros(Float64, branch_count, branch_count)
    for (i, br_id) in enumerate(branch)
        branch = ref[:branch][br_id]
        f_bus = bus_lookup[branch["f_bus"]]
        t_bus = bus_lookup[branch["t_bus"]]
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

    df_ptdf = DataFrame(ptdf_matrix, :auto)
    CSV.write(joinpath(output_dir, "$(case_name)_ptdf_matrix.csv"), df_ptdf)

    gen_count = length(gen)
    bus_gen_map = zeros(Int, bus_count, gen_count)
    for (i, gen_id) in enumerate(gen)
        gen_bus = ref[:gen][gen_id]["gen_bus"]
        bus_pos = bus_lookup[gen_bus]
        bus_gen_map[bus_pos, i] = 1
    end
    df_bus_gen_map = DataFrame(bus_gen_map, :auto)
    CSV.write(joinpath(output_dir, "$(case_name)_bus_gen_map.csv"), df_bus_gen_map)

    df_bus_ids = DataFrame(bus_id=bus)
    CSV.write(joinpath(output_dir, "$(case_name)_bus_ids.csv"), df_bus_ids)

    df_base_mva = DataFrame(parameter=["base_mva"], value=[ref[:baseMVA]])
    CSV.write(joinpath(output_dir, "$(case_name)_base_mva.csv"), df_base_mva)

    println("DC constraints saved to $output_dir")
end

function main(args=ARGS)
    usage = "Usage: julia --project=data_generalization data_generalization/export_dc_constraints.jl CASE_FILE OUTPUT_DIR"
    if args == ["--help"]
        println(usage)
        return
    end
    length(args) == 2 || error(usage)
    export_dc_constraints(args...)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    DCConstraints.main()
end
