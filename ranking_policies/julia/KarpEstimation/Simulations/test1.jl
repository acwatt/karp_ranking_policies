# This file is designed to create highly parallel simuluations and estimation of the model using Turing.
# For distriubted computing on Savio cluster
#! remove all in .julia/compiled/v1.9/
println(Pkg.project().path)
#! precompile here
using Distributed, SlurmClusterManager
addprocs(SlurmManager())  # this will only work when run on Savio
