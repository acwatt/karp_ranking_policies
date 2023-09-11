#!/usr/bin/env julia --threads=auto --project=./julia/KarpEstimation

using Distributed, SlurmClusterManager
# Command below for creating the package locally, then instantiating in remote nodes
# ]add Turing DataFrames Dates Optim ProgressMeter LoggingExtras CSV SMTPClient
using BenchmarkTools
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
# t = @benchmark distributed_test(); display(t)
# @everywhere println("hello from $(myid()):$(gethostname()) with $(Threads.nthreads()) threads")



# include("../KarpEstimation.jl")
