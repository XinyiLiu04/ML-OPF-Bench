using JSON3, PowerModels, JuMP, Ipopt
const MOI=JuMP.MOI
input_path, output_path=ARGS
ispath(output_path) && error("Refusing to overwrite diagnostics")
inputs=JSON3.read(read(input_path,String)); records=[]
for sample in inputs
    data=PowerModels.parse_file(String(sample.case_path))
    pd=Dict(zip(Int.(sample.load_ids),Float64.(sample.pd)))
    qd=Dict(zip(Int.(sample.load_ids),Float64.(sample.qd)))
    for load in values(data["load"])
        b=load["load_bus"]
        if haskey(pd,b); load["pd"]=pd[b]; load["qd"]=qd[b]; end
    end
    pm=PowerModels.instantiate_model(data,PowerModels.ACPPowerModel,PowerModels.build_opf;
        setting=Dict("output"=>Dict("duals"=>true)))
    result=PowerModels.optimize_model!(pm,optimizer=optimizer_with_attributes(Ipopt.Optimizer,"print_level"=>0,"sb"=>"yes"))
    result["termination_status"]==MOI.LOCALLY_SOLVED || error("Diagnostic solve failed")
    rec=Dict("case"=>sample.case,"scenario"=>sample.scenario,"index"=>sample.index,
        "status"=>string(result["termination_status"]),"objective"=>result["objective"],"solution"=>result["solution"])
    bounds=Dict()
    for key in (:pg,:qg,:vm)
        vars=PowerModels.var(pm,key)
        bounds[string(key)]=Dict(string(k)=>[dual(LowerBoundRef(vars[k])),dual(UpperBoundRef(vars[k]))] for k in keys(vars))
    end
    rec["bounds"]=bounds
    pv,qv=PowerModels.var(pm,:p),PowerModels.var(pm,:q)
    arcs=Dict(pv[k]=>k for k in axes(pv,1)); merge!(arcs,Dict(qv[k]=>k for k in axes(qv,1)))
    thermal=[]
    for con in all_constraints(pm.model,QuadExpr,MOI.LessThan{Float64})
        found=Set{Tuple{Int,Int,Int}}()
        for (_,a,b) in quad_terms(constraint_object(con).func)
            haskey(arcs,a) && push!(found,arcs[a]); haskey(arcs,b) && push!(found,arcs[b])
        end
        length(found)==1 || error("Ambiguous thermal constraint")
        push!(thermal,Dict("arc"=>collect(only(found)),"dual"=>dual(con)))
    end
    rec["thermal"]=thermal
    flow_bounds=[]
    for (kind,vars) in [("p",pv),("q",qv)]
        for arc in axes(vars,1)
            v=vars[arc]
            push!(flow_bounds,Dict("kind"=>kind,"arc"=>collect(arc),
                "lower"=>has_lower_bound(v) ? lower_bound(v) : nothing,
                "upper"=>has_upper_bound(v) ? upper_bound(v) : nothing,
                "dual_lower"=>has_lower_bound(v) ? dual(LowerBoundRef(v)) : 0.0,
                "dual_upper"=>has_upper_bound(v) ? dual(UpperBoundRef(v)) : 0.0))
        end
    end
    rec["flow_bounds"]=flow_bounds
    push!(records,rec)
end
open(output_path,"w") do io
    JSON3.write(io,Dict("note"=>"Four authorized diagnostic solves only; not replacement labels or timing benchmarks","records"=>records))
end
