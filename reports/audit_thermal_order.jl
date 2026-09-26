using PowerModels, JuMP, JSON3
const MOI = JuMP.MOI

case_path, output_path = ARGS
case = PowerModels.parse_file(case_path)
pm = PowerModels.instantiate_model(case, PowerModels.ACPPowerModel, PowerModels.build_opf)
pvars, qvars = PowerModels.var(pm, :p), PowerModels.var(pm, :q)
arcs = Dict(pvars[key] => key for key in axes(pvars, 1))
merge!(arcs, Dict(qvars[key] => key for key in axes(qvars, 1)))
constraints = JuMP.all_constraints(pm.model, JuMP.QuadExpr, MOI.LessThan{Float64})
order = []
for con in constraints
    found = Set{Tuple{Int,Int,Int}}()
    expr = JuMP.constraint_object(con).func
    for (_, a, b) in JuMP.quad_terms(expr)
        haskey(arcs, a) && push!(found, arcs[a])
        haskey(arcs, b) && push!(found, arcs[b])
    end
    length(found) == 1 || error("Thermal expression has ambiguous branch variables")
    push!(order, collect(only(found)))
end
ispath(output_path) && error("Audit output already exists")
open(output_path, "w") do io
    JSON3.write(io, Dict("case" => case_path, "constraint_arcs" => order,
                        "note" => "Model construction only; no optimization or labels generated"))
end
println("Recorded ", length(order), " thermal constraints without solving")
