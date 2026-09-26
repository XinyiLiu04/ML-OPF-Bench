using JSON3, JuMP, Ipopt, PowerModels

input = JSON3.read(read(ARGS[1], String))
optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "sb" => "yes")

function ac_solve(sample, template)
    data = deepcopy(template)
    pd = Dict(zip(input.load_bus_ids, sample.pd))
    qd = Dict(zip(input.load_bus_ids, sample.qd))
    for load in values(data["load"])
        bus = load["load_bus"]
        if haskey(pd, bus)
            load["pd"], load["qd"] = pd[bus], qd[bus]
        end
    end
    result = solve_ac_opf(data, optimizer;
                          setting=Dict("output" => Dict("branch_flows" => true)))
    return string(result["termination_status"]), result["objective"]
end

function dc_solve(sample, c)
    model = Model(optimizer)
    set_silent(model)
    ng = length(c.pg_min)
    H = reduce(vcat, permutedims.(Vector{Float64}.(c.ptdf)))
    M = reduce(vcat, permutedims.(Vector{Float64}.(c.gen_bus_map)))
    pd = Vector{Float64}(sample.pd)
    @variable(model, c.pg_min[g] <= pg[g=1:ng] <= c.pg_max[g])
    @objective(model, Min, sum(c.cost_c2[g] * pg[g]^2 + c.cost_c1[g] * pg[g] + c.cost_c0[g] for g=1:ng))
    @constraint(model, sum(pg) == sum(pd))
    flow = H * (M * pg - pd)
    for line in eachindex(c.rate_a)
        if c.rate_a[line] < 1e10
            @constraint(model, -c.rate_a[line] <= flow[line] <= c.rate_a[line])
        end
    end
    optimize!(model)
    return string(termination_status(model)), has_values(model) ? objective_value(model) : nothing
end

template = input.formulation == "ac" ? PowerModels.parse_file(String(input.case_path)) : input.constraints
solve = input.formulation == "ac" ? ac_solve : dc_solve
solve(input.samples[1], template)  # Exclude Julia compilation from latency.
records = []
for sample in input.samples
    start = time_ns()
    status, cost = solve(sample, template)
    elapsed = (time_ns() - start) / 1e6
    push!(records, (; index=sample.index, milliseconds=elapsed, status, cost))
end
isfile(ARGS[2]) && error("Refusing to overwrite an existing result")
open(ARGS[2], "w") do file
    JSON3.write(file, (; solver="Ipopt", formulation=input.formulation, records,
                       julia_version=string(VERSION), scope="sample preparation, model construction and solve; warm compilation"))
end
