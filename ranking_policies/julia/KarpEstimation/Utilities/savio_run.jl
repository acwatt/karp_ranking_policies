#!/usr/bin/env julia --threads=auto --project=./julia/KarpEstimation

using Distributed, SlurmClusterManager
# Command below for creating the package locally, then instantiating in remote nodes
# ]add Turing DataFrames Dates Optim ProgressMeter LoggingExtras CSV SMTPClient
addprocs(SlurmManager())
# instantiate and precompile environment in all processes
@everywhere begin
    using Pkg; Pkg.activate(@__DIR__)
    Pkg.instantiate(); Pkg.precompile()
end
