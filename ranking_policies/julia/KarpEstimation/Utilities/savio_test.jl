#!/usr/bin/env julia --threads=auto --project=./julia/KarpEstimation
# Test savio


# using Distributed

# @everywhere begin
#     project_env = joinpath(splitpath(@__DIR__)[1:end-1])
#     using Pkg; Pkg.activate(project_env)
#     Pkg.instantiate(); Pkg.precompile()
# end

# @everywhere begin
#     # import Pkg; Pkg.add("BenchmarkTools")
#     using BenchmarkTools
#     using ProgressMeter
#     # Test parallel computing using Threads.@threads
#     # @benchmark sort(data) setup=(data=rand(10))

#     println("# of Threads:$(Threads.nthreads())")

#     function parallel_test()
#         Threads.@threads for i in 1:100
#             sleep(1/100)
#         end
#     end
# end



# t = @benchmark parallel_test()
# t = @benchmark distributed_test(); display(t)

using Distributed, SlurmClusterManager
# using BenchmarkTools
addprocs(SlurmManager())
# instantiate and precompile environment in all processes
@everywhere begin
    project_env = joinpath(splitpath(@__DIR__)[1:end-1])
    println(project_env)
    using Pkg; Pkg.activate(project_env)
    Pkg.instantiate(); Pkg.precompile()
end
@everywhere begin
    using Distributions
end

function distributed_test() 
    pmap(1:100) do i
        try
            sleep(1/100)
            true # success
        catch e
            false # failure
        end
    end
    return nothing
end

println("# of Threads:$(Threads.nthreads())")
println("# of workers:$(nprocs())")
@info "# of Threads:$(Threads.nthreads())"
@info "# of workers:$(nprocs())"
# t = @benchmark distributed_test(); display(t)
# @everywhere println("hello from $(myid()):$(gethostname()) with $(Threads.nthreads()) threads")



# include("../KarpEstimation.jl")
