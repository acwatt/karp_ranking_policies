#!/usr/bin/env julia --threads=auto
using Pkg
Pkg.status()
using Distributed, SlurmClusterManager
addprocs(SlurmManager())

# Remote environment
rm(string(@__DIR__, "/Manifest.toml"))
rm(string(@__DIR__, "/Project.toml"))
@info "Done removing environment files."

# Resetup the environment files
Pkg.activate(@__DIR__)
Pkg.add(["SMTPClient", "CSV", "Turing", "Optim", "DynamicHMC", "Bijectors"])
# Pkg.resolve()
# Pkg.precompile()
@info "Done creating new environment."

# instantiate/add environment packages in all processes
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.add(["SMTPClient", "CSV", "Turing", "Optim", "DynamicHMC", "Bijectors"])
    # include("Communications.jl")  # send_txt
end
@info "Done with project activation."

# Wait until all processes have activated their projects, then add packages
# @everywhere begin
#     Pkg.add(["SMTPClient", "CSV", "Turing", "Optim", "DynamicHMC", "Bijectors"])
# end
# @info "Done with package installs."

# Wait until all packages have been added, then load needed packages
@everywhere begin
    include("Communications.jl")  # send_txt
end
# include("Communications.jl")  # send_txt
# send_txt("savio_test start", "")
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

