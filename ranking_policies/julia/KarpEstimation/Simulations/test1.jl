# This file is designed to create highly parallel simuluations and estimation of the model using Turing.
# For distriubted computing on Savio cluster
#! remove all in .julia/compiled/v1.9/
using Pkg
println(Pkg.project().path)
#! precompile here
using Distributed, SlurmClusterManager
addprocs(SlurmManager())  # this will only work when run on Savio

Pkg.add(["Turing", "NamedArrays"])
using Turing
@info "using Turing in global"

@everywhere using Turing
@info "using Turing everywhere"
