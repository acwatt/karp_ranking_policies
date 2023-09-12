using Distributed, SlurmClusterManager
addprocs(SlurmManager())

# Remove environment
try; rm(string(@__DIR__, "/Manifest.toml")); catch e; end
try; rm(string(@__DIR__, "/Project.toml")); catch e; end
@info "Done removing environment files."

# Resetup the environment files
using Pkg
Pkg.activate(@__DIR__)
Pkg.add(["SMTPClient", "CSV", "Turing", "Optim", "DynamicHMC", "Bijectors"])
@info "Done creating new environment."

# instantiate/add environment in all processes
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
    # include("Communications.jl")  # send_txt
end
@info "Done with project activation."

# Wait until all packages have been added, then load needed packages
# Include files in global scope first, then everywhere according to https://github.com/JuliaLang/julia/issues/16788#issuecomment-226977022
include("Communications.jl")  # send_txt
@everywhere begin
    include("Communications.jl")  # send_txt
end
send_txt("savio_test start", "")
@info "Done with package loads."

# Test distributed computing and txting updates
function distributed_test() 
    pmap(1:100) do i
        try
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
distributed_test()
@info "Done with distributed_test."
# send_txt("savio_test end", "")

