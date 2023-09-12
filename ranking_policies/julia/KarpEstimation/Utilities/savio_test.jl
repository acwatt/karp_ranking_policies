using Distributed, SlurmClusterManager
addprocs(SlurmManager())

# Remove environment
try; rm(string(@__DIR__, "/Manifest.toml")); catch e; end
try; rm(string(@__DIR__, "/Project.toml")); catch e; end
@info "Done removing environment files."

# Resetup the environment files
using Pkg
Pkg.activate(@__DIR__)
Pkg.add(["SMTPClient", "CSV", "Turing", "Optim", "DynamicHMC", "Bijectors", "DataFrames"])
@info "Done creating new environment."

# instantiate/add environment in all processes
@everywhere begin
    using Pkg
    function act_int(;i=1)
        try
            Pkg.activate(@__DIR__)
            Pkg.instantiate()
        catch e
            println("Error $i: ", e)
            act_int(;i=i+1)
        end
    end
    act_int()
    # include("Communications.jl")  # send_txt
end
@info "Done with project activation."

# Wait until all packages have been added, then load needed packages
# Include files in global scope first, then everywhere according to https://github.com/JuliaLang/julia/issues/16788#issuecomment-226977022
using DataFrames
include("Communications.jl")  # send_txt
@everywhere begin
    using DataFrames
    include("Communications.jl")  # send_txt
end
send_txt("savio_test start", "")
@info "Done with package loads."


# Test distributed computing and txting updates
function distributed_test!(df) 
    pmap(1:100) do i
        try
            println("a = ", df[i, :a], ", b = ", df[i, :b], ", i = ", i, "worker = ", myid())
            df[i, :c] = df[i, :a] + df[i, :b]
            sleep(1/100)
            true # success
        catch e
            false # failure
        end
        i%10 == 0 ? send_txt("savio_test update", "i = $i") : nothing
    end
    return nothing
end
# t = @benchmark distributed_test(); display(t)
df = DataFrame(a = 1:100, b = 101:200, c = 201:300)
distributed_test!(df)
@show df

@info "Done with distributed_test."
# send_txt("savio_test end", "")

