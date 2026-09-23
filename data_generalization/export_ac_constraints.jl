module ACConstraints

using PowerModels, DataFrames, CSV

function export_ac_constraints(case_path::String, output_dir::String)
    ispath(output_dir) && error("Output path already exists: $output_dir")
    mkpath(output_dir)
    case_name = splitext(basename(case_path))[1]

    data = PowerModels.parse_file(case_path)
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]

    println("Parsed case: $(case_name)")

    base_mva = ref[:baseMVA]
    df_base_mva = DataFrame(parameter=["baseMVA"], value=[base_mva])
    CSV.write(joinpath(output_dir, "$(case_name)_base_mva.csv"), df_base_mva)

    bus_ids = sort(collect(keys(ref[:bus])))

    load_pd = Dict(id => 0.0 for id in bus_ids)
    load_qd = Dict(id => 0.0 for id in bus_ids)

    for (_, load) in ref[:load]
        if get(load, "status", 1) == 1
            b = load["load_bus"]
            load_pd[b] += get(load, "pd", 0.0)
            load_qd[b] += get(load, "qd", 0.0)
        end
    end

    bus_data = DataFrame(
        bus_id  = Int[],
        type    = Int[],
        pd_pu   = Float64[],
        qd_pu   = Float64[],
        vmin_pu = Float64[],
        vmax_pu = Float64[],
        vm_pu   = Float64[],
        va_deg  = Float64[],
        base_kv = Float64[]
    )

    for id in bus_ids
        bus = ref[:bus][id]
        push!(bus_data, [
            id,
            bus["bus_type"],
            load_pd[id],
            load_qd[id],
            bus["vmin"],
            bus["vmax"],
            get(bus, "vm", 1.0),
            get(bus, "va", 0.0),
            get(bus, "base_kv", 110.0)
        ])
    end
    CSV.write(joinpath(output_dir, "$(case_name)_bus_data.csv"), bus_data)

    gen_ids = sort(collect(keys(ref[:gen])))
    gen_data = DataFrame(
        gen_id      = Int[],
        bus_id      = Int[],
        pg_min_pu   = Float64[],
        pg_max_pu   = Float64[],
        qg_min_pu   = Float64[],
        qg_max_pu   = Float64[],
        vg_pu       = Float64[],
        cost_c2     = Float64[],
        cost_c1     = Float64[],
        cost_c0     = Float64[]
    )

    for id in gen_ids
        g = ref[:gen][id]

        coeffs = get(g, "cost", Float64[])
        c2, c1, c0 = 0.0, 0.0, 0.0
        if length(coeffs) == 3
            c2, c1, c0 = coeffs[1], coeffs[2], coeffs[3]
        elseif length(coeffs) == 2
            c1, c0 = coeffs[1], coeffs[2]
        end

        push!(gen_data, [
            id,
            g["gen_bus"],
            g["pmin"],
            g["pmax"],
            g["qmin"],
            g["qmax"],
            g["vg"],
            c2, c1, c0
        ])
    end
    CSV.write(joinpath(output_dir, "$(case_name)_gen_data.csv"), gen_data)

    n_buses = length(bus_ids)
    n_gen = length(gen_ids)
    bus_id_to_idx = Dict(bus_id => idx for (idx, bus_id) in enumerate(bus_ids))

    bus_gen_matrix = zeros(Int, n_buses, n_gen)
    for (gen_idx, gen_id) in enumerate(gen_ids)
        gen_bus_id = ref[:gen][gen_id]["gen_bus"]
        bus_idx = bus_id_to_idx[gen_bus_id]
        bus_gen_matrix[bus_idx, gen_idx] = 1
    end

    gen_col_names = ["gen_$i" for i in 1:n_gen]
    bus_gen_df = DataFrame(bus_gen_matrix, gen_col_names)
    insertcols!(bus_gen_df, 1, :bus_id => bus_ids)
    CSV.write(joinpath(output_dir, "$(case_name)_bus_gen_map.csv"), bus_gen_df)

    branch_ids = sort(collect(keys(ref[:branch])))
    branch_data = DataFrame(
        branch_id = Int[],
        f_bus     = Int[],
        t_bus     = Int[],
        r_pu      = Float64[],
        x_pu      = Float64[],
        b_pu      = Float64[],
        rate_a_pu = Float64[],
        tap_ratio = Float64[],
        shift_deg = Float64[]
    )

    for id in branch_ids
        br = ref[:branch][id]

        rate_a_raw = get(br, "rate_a", Inf)

        rate_a_val = isfinite(rate_a_raw) ? rate_a_raw : 0.0

        push!(branch_data, [
            id,
            br["f_bus"], br["t_bus"],
            br["br_r"], br["br_x"],
            get(br, "b_fr", 0.0) + get(br, "b_to", 0.0),
            rate_a_val,
            get(br, "tap", 1.0),
            get(br, "shift", 0.0)
        ])
    end
    CSV.write(joinpath(output_dir, "$(case_name)_branch_data.csv"), branch_data)

    slack_buses = DataFrame(bus_id = collect(keys(ref[:ref_buses])))
    CSV.write(joinpath(output_dir, "$(case_name)_slack_buses.csv"), slack_buses)

    println("Done (p.u. units). Files saved to: $(output_dir)")
end

function main(args=ARGS)
    usage = "Usage: julia --project=data_generalization data_generalization/export_ac_constraints.jl CASE_FILE OUTPUT_DIR"
    if args == ["--help"]
        println(usage)
        return
    end
    length(args) == 2 || error(usage)
    export_ac_constraints(args...)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    ACConstraints.main()
end
