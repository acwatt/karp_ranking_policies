"""
module: Estimators.jl
description: defines estimators for ranking policies project
author: Aaron C Watt (UCB Grad Student, Ag & Resource Econ)
email: aaron@acwatt.net
created: 2023-06-13
last update: 
See docs/code/Estimators.md for more notes
"""
module Estimators
# Import dependencies
using Optim


############ Define specific estimators ############
# Uses: 
include("EstimatorsAR1.jl")
# Uses: Optim
include("EstimatorsMLE.jl")
include("EstimatorsGMM.jl")



end