#==============================================================================
file: covariance_matrix.jl
description: Main file for project. 
    Contains: model functions, estimation algorithms, real data loading
author: Aaron C Watt (UCB Grad Student, Ag & Resource Econ)
email: aaron@acwatt.net
created: 2022-01-02
last update: 2023-06-05
See notes section below for more details.
==============================================================================#

# Loading packages can often take a long time, but Julia is very fast once packages are loaded
# using Zygote  # for reverse-mode auto-differentiation
using Random  # Random numbers
using Distributions
using DataFrames
using DataFramesMeta
using CategoricalArrays
using LinearAlgebra  # Diagonal
using StatsModels  # formulas and modelmatrix
using StatsPlots
using GLM  # OLS (lm) and GLS (glm)
using Optim  # optimize (minimizing)
using FiniteDifferences  # numerical gradient
# using Symbolics  # symbolic functions
import Dates
using Latexify  # output symbolic expressions as latex code
# using ModelingToolkit  # more symbolic matrix functions
# using Reduce  # more symbolic matrix functions (specifically Reduce.Algebra.^)
# @force using Reduce.Algebra  # to extend native functions to Sybmol/Expr types
using Logging
import CSV
using ProgressMeter

import Revise
Revise.includet("reg_utils.jl")  # mygls()


#=
]add Random Distributions DataFrames DataFramesMeta CategoricalArrays LinearAlgebra StatsModels StatsPlots GLM Optim FiniteDifferences Dates Latexify Logging CSV Revise
=#

"""
NOTES:
Purpose: Estimate the parameters ρ, σₐ², σᵤ² of Larry Karp's emissions
    model using iterated Feasible GLS and MLE

How: Most of this script defines functions that represent equations from
    the model. Finally, in the "Iterate to convergence!" section at the
    end, the equations are used to iteratively estimate:
    (1) the GLS parameters b₀ᵢ and β (time trend) as inputs to (2)
    (2) the MLE parameters ρ, σₐ², σᵤ²

First, I test this on generated data that we know the parameter values
    for to see how close the estimates are to the "true" values. I test
    the estimates using different values of T, N, σₐ², σᵤ².

Lastly, I (still need to) estimate the paramters on the actual data that
    Larry provided.

Time to run: Most of the functions at the end that actually run the
    estimations also have time estimates in a comment after them. These
    were run on a linux computer with 32 GB of RAM and 
    AMD Ryzen 3 2300x quad-core processor. Times may be more or less
    depending on your system. Julia does multithreading automatically,
    meaning it will take up all available CPU during most compuations
    until the script has complete.

Last update: 2022-12-15
"""

#######################################################################
#           Setup
#######################################################################
io = open("log.txt", "a")
logger = SimpleLogger(io)
# debuglogger = ConsoleLogger(stderr, Logging.Debug)
global_logger(logger)
s = "\n"^20 * "="^60 * "\nCOVARIANCE MATRIX ESTIMATION BEGIN\n" * "="^60
@info s datetime=Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
num2country = Dict(1=>"USA", 2=>"EU", 3=>"BRIC", 4=>"Other")
country2num = Dict("USA"=>1, "EU"=>2, "BRIC"=>3, "Other"=>4)




#######################################################################
#           Equation Functions
#######################################################################


"""
Function to fill the NT x NT matrix with elements defined by ijs_func
"""
function fill_matrix(ijs_func, ρ, σₐ², σᵤ², N, T; verbose = false)
    # Initalize matrix of 0s
    mat = zeros(N * T, N * T)

    # Fill in upper triangle
    idx = [(i, j) for i ∈ 1:N*T for j ∈ i:N*T]
    for (row, col) in idx
        t = Integer(ceil(row / N))
        i = row - (t - 1) * N

        τ = Integer(ceil(col / N))
        s = τ - t
        j = col - (τ - 1) * N

        mat[row, col] = ijs_func(ρ, σₐ², σᵤ², i, j, s, N)
    end

    # Fill in lower triangle by symmetry
    mat = Symmetric(mat)
    if verbose
        println("\nFull NT x NT Matirx:")
        for r in 1:size(mat)[1]
            println(round.(mat[r, :]; sigdigits = 4))
        end
    end
    return (mat)
end


# Negative Log Likelihood (want to minimize)
function nLL(ρ, σₐ², σᵤ², v, N, T)
    V = Σ(ρ,σₐ²,σᵤ²,N,T)
    nLL = (1/2) * ( v'*V^-1*v + log(det(big.(V))) )
    nLL = Float64(nLL)
    return(nLL)
end
function nLL(θ, v, N, T)
    V = Σ(θ.ρ, θ.σₐ², θ.σᵤ², N, T)
    nLL = (1/2) * ( v'*V^-1*v + log(det(big.(V))) )
    nLL = Float64(nLL)
    return(nLL)
end

function ∂nLL∂y(y, ρ, σₐ², σᵤ², v, N, T)
    ijs_fun(ρ, σₐ², σᵤ², i, j, s, N) = ∂Σ∂y(y, ρ, σₐ², σᵤ², i, j, s, N)
    ∂V∂y = fill_matrix(ijs_fun, ρ, σₐ², σᵤ², N, T)

    V = Σ(ρ,σₐ²,σᵤ²,N,T)
    vV⁻¹∂V∂yV⁻¹v = v'*V^-1*∂V∂y*V^-1*v
    V⁻¹∂V∂y = V^-1*∂V∂y

    ∂LL∂y = -1/2 * (-vV⁻¹∂V∂yV⁻¹v + tr(V⁻¹∂V∂y))
    return(-∂LL∂y)
end


# Define the gradient of the negative log likelihood
function nLL_grad!(gradient_vec, params, v, N, T)
    ρ = params[1]; σₐ² = params[2]; σᵤ² = params[3]
    gradient_vec[1] = ∂nLL∂y("ρ", ρ, σₐ², σᵤ², v, N, T)
    gradient_vec[2] = ∂nLL∂y("σₐ²", ρ, σₐ², σᵤ², v, N, T)
    gradient_vec[3] = ∂nLL∂y("σᵤ²", ρ, σₐ², σᵤ², v, N, T)
    return gradient_vec
end
function nLL_grad2(gradient_vec, params, ρ, v, N, T)
    σₐ² = params[1]; σᵤ² = params[2]
    gradient_vec[1] = ∂nLL∂y("σₐ²", ρ, σₐ², σᵤ², v, N, T)*N
    gradient_vec[2] = ∂nLL∂y("σᵤ²", ρ, σₐ², σᵤ², v, N, T)
    return gradient_vec
end

""" """
function nLL_grad_fidiff!(gradient_vec, ρσₐ²σᵤ², v, N, T)
    g = FiniteDifferences.grad(central_fdm(5, 1),
                           ρσₐ²σᵤ² -> nLL(ρσₐ²σᵤ²[1], ρσₐ²σᵤ²[2], ρσₐ²σᵤ²[3], v, N, T),
                           ρσₐ²σᵤ²)
    # update the gradient vector
    gradient_vec[1] = g[1][1]  # ρ
    gradient_vec[2] = g[1][2]  # σₐ²
    gradient_vec[3] = g[1][3]  # σᵤ²
    return gradient_vec
end
function nLL_grad_fidiff2!(gradient_vec, σₐ²σᵤ², ρ, v, N, T)
    g = FiniteDifferences.grad(central_fdm(5, 1),
                           σₐ²σᵤ² -> nLL(ρ, σₐ²σᵤ²[1], σₐ²σᵤ²[2], v, N, T),
                           σₐ²σᵤ²)
    # update the gradient vector
    gradient_vec[1] = g[1][1]  # σₐ²
    gradient_vec[2] = g[1][2]  # σᵤ²
    return gradient_vec
end



METHOD_MAP = Dict("gradient decent" => GradientDescent(),
                  "conjugate gradient" => ConjugateGradient(),
                  "BFGS" => BFGS(),
                  "LBFGS" => LBFGS(),
                  "momentum gradient" => MomentumGradientDescent(),
                  "accelerated gradient" => AcceleratedGradientDescent()
)

"""
    mymle(ρstart, σₐ²start, σᵤ²start, v)

    Return likelihood-maximizing values of ρ, σₐ², and σᵤ².
    Given vector of residuals, minimize the 
    @param ρstart: float, starting value of ρ
    @param σₐ²start: float, starting value of σₐ²
    @param σᵤ²start: float, starting value of σᵤ²
    @param v: Nx1 vector of residuals
    @param N: int>1, number of units/regions
    @param T: int>1, number of time periods
    @param lower: float 3-vector, lower bounds for parameters ρ, σₐ², σᵤ²
    @param upper: float 3-vector, upper bounds for parameters ρ, σₐ², σᵤ²
"""
function mymle_3(ρstart, σₐ²start, σᵤ²start, v, N, T;
    lower = [1e-4, 1e-4, 1e-4], upper = [1 - 1e-4, Inf, Inf], analytical = true, method = "gradient decent",
    show_trace=false)
    println("Starting MLE with $ρstart, $σₐ²start, $σᵤ²start")
    @info "mymle_3()" ρstart σₐ²start σᵤ²start N T lower upper analytical method
    # Starting paramerter values
    params0 = [ρstart, σₐ²start, σᵤ²start]

    # Function of only parameters (given residuals v)
    objective(ρσₐ²σᵤ²) = nLL(ρσₐ²σᵤ²[1], ρσₐ²σᵤ²[2], ρσₐ²σᵤ²[3], v, N, T)
    objective2(σₐ²σᵤ²) = nLL(ρstart, σₐ²σᵤ²[1], σₐ²σᵤ²[2], v, N, T)

    # Minimize over parameters
    # 
    println("Analytical gradient: ", nLL_grad!([0.,0.,0.], params0, v, N, T))  # with estimating ρ
    # println("Analytical gradient: ", nLL_grad2([0.,0.], params0[2:3], ρstart, v, N, T))  # without estimating ρ

    
    # println("Fin. Diff. gradient: ", nLL_grad_fidiff!([0.,0.,0.], params0, v, N, T))
    # println("Fin. Diff. gradient: ", nLL_grad_fidiff2!([0.,0.], params0[2:3], ρstart, v, N, T))

    # Calculates gradient with no parameter bounds
    # optimum = optimize(objective, params0, LBFGS())
    # Results in a domain error in the log function because the gradient pushes
    #   it to a negative value inside the log

    # Analytical gradient with no parameter bounds
    # optimum = optimize(objective, grad!, params0, LBFGS())
    # Results in a domain error in the log function because the gradient pushes
    #   it to a negative value inside the log

    if analytical   
        # Analytical gradient with parameter bounds
        println("Using analytical gradient.")
        grad!(storage, x) = nLL_grad!(storage, x, v, N, T)  # estimating ρσₐ²σᵤ²  # with estimating ρ
        optimum = optimize(objective, grad!, 
                           lower, upper, params0, 
                           Fminbox(method_map[method]),
                           Optim.Options(show_trace = true, 
                                         show_every=1, 
                                         g_tol = 1e-8,
                                         time_limit = 10, 
                                         extended_trace=true))  # with estimating ρ
        # grad2!(storage, x) = nLL_grad2(storage, x, ρstart, v, N, T)  # estimating σₐ²σᵤ²  # without estimating ρ
        # optimum = optimize(objective2, grad2!, lower[2:3], upper[2:3], params0[2:3], Fminbox(method_map[method]),  # without estimating ρ
        #                    Optim.Options(show_trace = show_trace, show_every=1, g_tol = 1e-8, time_limit = 1))
    else
        # Calculates gradient with parameter bounds
        println("Using estimated gradient.")
        println("Fin. Diff. gradient: ", nLL_grad_fidiff2!([0.,0.], params0[2:3], ρstart, v, N, T))
        optimum = optimize(objective, lower, upper, params0, Fminbox(method_map[method]),  # with estimating ρ
                           Optim.Options(show_trace = show_trace, show_every=5))
        # grad2!(storage, x) = nLL_grad_fidiff2!(storage, x, v, N, T)
        # optimum = optimize(objective2, lower[2:3], upper[2:3], params0[2:3], Fminbox(method_map[method]),  # without estimating ρ
        #                    Optim.Options(show_trace = show_trace, show_every=1))
    end 
    
    # Return the values
    LL = -optimum.minimum
    ρ, σₐ², σᵤ² = optimum.minimizer                # with estimating ρ
    # ρ, σₐ², σᵤ² = ρstart, optimum.minimizer...   # without estimating ρ
    return (ρ, σₐ², σᵤ², LL)
end

"""
Same MLE estimation as above, but assuming value for rho as given
Only estimates two sigma values
Have not yet editied -- would require editing nLL and gradients
"""
function mymle_2(ρstart, σₐ²start, σᵤ²start, v, N, T;
    lower = [1e-4, 1e-4, 1e-4], upper = [1 - 1e-4, Inf, Inf], analytical = true, method = "gradient decent",
    show_trace=false)
    println("Starting MLE with $ρstart, $σₐ²start, $σᵤ²start")
    @info "mymle_2()" ρstart σₐ²start σᵤ²start lower upper analytical method
    # Starting paramerter values
    params0 = [ρstart, σₐ²start, σᵤ²start]

    # Function of only parameters (given residuals v)
    objective2(σₐ²σᵤ²) = nLL(ρstart, σₐ²σᵤ²[1], σₐ²σᵤ²[2], v, N, T)

    # Minimize over parameters
    # println("Analytical gradient: ", nLL_grad2([0.,0.], params0[2:3], ρstart, v, N, T))
    # println("Fin. Diff. gradient: ", nLL_grad_fidiff2!([0.,0.], params0[2:3], ρstart, v, N, T))

    # Calculates gradient with no parameter bounds
    # optimum = optimize(objective, params0, LBFGS())
    # Results in a domain error in the log function because the gradient pushes
    #   it to a negative value inside the log

    # Analytical gradient with no parameter bounds
    # optimum = optimize(objective, grad!, params0, LBFGS())
    # Results in a domain error in the log function because the gradient pushes
    #   it to a negative value inside the log

    if analytical   
        # Analytical gradient with parameter bounds
        grad2!(storage, x) = nLL_grad2(storage, x, ρstart, v, N, T)  # estimating σₐ²σᵤ²
        optimum = optimize(objective2, grad2!, lower[2:3], upper[2:3], params0[2:3], Fminbox(METHOD_MAP[method]),
                           Optim.Options(show_trace = show_trace, show_every=1, g_tol = 1e-8, time_limit = 1))
    else
        # Calculates gradient with parameter bounds
        optimum = optimize(objective2, lower[2:3], upper[2:3], params0[2:3], Fminbox(METHOD_MAP[method]),
                           Optim.Options(show_trace = show_trace, show_every=1))
    end 

    # Return the values
    LL = -optimum.minimum
    # ρ, σₐ², σᵤ² = optimum.minimizer
    ρ, σₐ², σᵤ² = ρstart, optimum.minimizer...
    return (ρ, σₐ², σᵤ², LL)
end

function mymle_2(ρstart, σₐ²start, σᵤ²start, v, N, T;
                θLB = (ρ=0.878, σₐ²=1e-4, σᵤ²=1e-4), θUB = (ρ=0.879, σₐ²=Inf, σᵤ²=Inf), 
                analytical = true, method = "gradient decent", show_trace=false
                )
    println("Starting MLE with $ρstart, $σₐ²start, $σᵤ²start")
    @info "mymle_2()" ρstart σₐ²start σᵤ²start θLB θUB analytical method

    # Function of only parameters (given residuals v)
    objective2(σₐ²σᵤ²) = nLL(ρstart, σₐ²σᵤ²[1], σₐ²σᵤ²[2], v, N, T)

    if analytical   
        # Analytical gradient with parameter bounds
        grad2!(storage, x) = nLL_grad2(storage, x, ρstart, v, N, T)  # estimating σₐ²σᵤ²
        optimum = optimize(objective2, grad2!, [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [σₐ²start, σᵤ²start], Fminbox(METHOD_MAP[method]),
                           Optim.Options(show_trace = show_trace, show_every=1, g_tol = 1e-8, time_limit = 1))
    else
        # Calculates gradient with parameter bounds
        optimum = optimize(objective2, [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [σₐ²start, σᵤ²start], Fminbox(METHOD_MAP[method]),
                           Optim.Options(show_trace = show_trace, show_every=1))
    end 

    # Return the values
    LL = -optimum.minimum
    # ρ, σₐ², σᵤ² = optimum.minimizer
    ρ, σₐ², σᵤ² = ρstart, optimum.minimizer...
    return (ρ, σₐ², σᵤ², LL)
end


function mymle_2(θ₀, v, N, T;
                θLB = (ρ=0.878, σₐ²=1e-4, σᵤ²=1e-4), θUB = (ρ=0.879, σₐ²=Inf, σᵤ²=Inf), 
                analytical = true, method = "gradient decent", show_trace=false
                )
    println("Starting MLE with $ρstart, $σₐ²start, $σᵤ²start")
    @info "mymle_2()" θ₀ θLB θUB analytical method

    # Function of only parameters (given residuals v)
    objective2(σₐ²σᵤ²) = nLL(θ₀.ρ, σₐ²σᵤ²[1], σₐ²σᵤ²[2], v, N, T)

    if analytical   
        # Analytical gradient with parameter bounds
        grad2!(storage, σₐ²σᵤ²) = nLL_grad2(storage, σₐ²σᵤ², θ₀.ρ, v, N, T)  # estimating σₐ²σᵤ²
        optimum = optimize(objective2, grad2!, [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [θ₀.σₐ², θ₀.σᵤ²], Fminbox(METHOD_MAP[method]),
                           Optim.Options(show_trace = show_trace, show_every=1, g_tol = 1e-8, time_limit = 1))
        
    else
        # Calculates gradient with parameter bounds
        optimum = optimize(objective2, [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [σₐ²start, σᵤ²start], Fminbox(METHOD_MAP[method]),
                           Optim.Options(show_trace = show_trace, show_every=1))
    end 

    # Return the values
    LL = -optimum.minimum
    # ρ, σₐ², σᵤ² = optimum.minimizer
    ρ, σₐ², σᵤ² = ρstart, optimum.minimizer...
    return (ρ, σₐ², σᵤ², LL)
end

function mymle_2_testing_optim_algos(θ₀, v, N, T;
                θLB = (ρ=0.878, σₐ²=1e-4, σᵤ²=1e-4), θUB = (ρ=0.879, σₐ²=Inf, σᵤ²=Inf), 
                analytical = true, method = "gradient decent", show_trace=false
                )
    println("Starting MLE with $ρstart, $σₐ²start, $σᵤ²start")
    @info "mymle_2()" θ₀ θLB θUB analytical method

    # Function of only parameters (given residuals v)
    objective2(σₐ²σᵤ²) = nLL(θ₀.ρ, σₐ²σᵤ²[1], σₐ²σᵤ²[2], v, N, T)
    LL_2params(σₐ²σᵤ²) = -nLL(θ₀.ρ, σₐ²σᵤ²[1], σₐ²σᵤ²[2], v, N, T)

    if analytical   
        # Analytical gradient with parameter bounds
        grad2!(storage, σₐ²σᵤ²) = nLL_grad2(storage, σₐ²σᵤ², θ₀.ρ, v, N, T)  # estimating σₐ²σᵤ²
        optimum = optimize(objective2, grad2!, [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [θ₀.σₐ², θ₀.σᵤ²], Fminbox(METHOD_MAP[method]),
                           Optim.Options(show_trace = show_trace, show_every=1, g_tol = 1e-8, time_limit = 1))

        optimum1 = optimize(objective2, grad2!, [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [θ₀.σₐ², θ₀.σᵤ²], Fminbox(METHOD_MAP["LBFGS"]),
                           Optim.Options(show_trace = true, show_every=1, g_tol = 1e-8))

        optimum2 = optimize(objective2, grad2!, [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [θ₀.σₐ², θ₀.σᵤ²], Fminbox(METHOD_MAP["conjugate gradient"]),
                           Optim.Options(show_trace = true, show_every=1, g_tol = 1e-8))

        optimum3 = optimize(objective2, grad2!, [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [θ₀.σₐ², θ₀.σᵤ²], Fminbox(METHOD_MAP["gradient decent"]),
                           Optim.Options(show_trace = true, show_every=1, g_tol = 1e-8))

        optimum4 = optimize(objective2, grad2!, [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [θ₀.σₐ², θ₀.σᵤ²], Fminbox(METHOD_MAP["BFGS"]),
                           Optim.Options(show_trace = true, show_every=1, g_tol = 1e-8))

        optimum5 = optimize(objective2, grad2!, [θ₀.σₐ², θ₀.σᵤ²], METHOD_MAP["momentum gradient"],
                           Optim.Options(show_trace = true, show_every=1, g_tol = 1e-8))

        optimum6 = optimize(objective2, grad2!, [θ₀.σₐ², θ₀.σᵤ²], METHOD_MAP["accelerated gradient"],
                           Optim.Options(show_trace = true, show_every=1, g_tol = 1e-8))
        
        println("\nLBFGS:")
        println(optimum1.minimizer)
        println(optimum1.minimizer .- [θ.σₐ², θ.σᵤ²])
        println("\nconjugate gradient:")
        println(optimum2.minimizer)
        println(optimum2.minimizer .- [θ.σₐ², θ.σᵤ²])
        println("\ngradient decent:")
        println(optimum3.minimizer)
        println(optimum3.minimizer .- [θ.σₐ², θ.σᵤ²])
        println("\nBFGS:")
        println(optimum4.minimizer)
        println(optimum4.minimizer .- [θ.σₐ², θ.σᵤ²])
        println("\nmomentum gradient:")
        println(optimum5.minimizer)
        println(optimum5.minimizer .- [θ.σₐ², θ.σᵤ²])
        println("\naccelerated gradient:")
        println(optimum6.minimizer)
        println(optimum6.minimizer .- [θ.σₐ², θ.σᵤ²])
        
    else
        # Calculates gradient with parameter bounds
        optimum = optimize(objective2, [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [σₐ²start, σᵤ²start], Fminbox(METHOD_MAP[method]),
                           Optim.Options(show_trace = show_trace, show_every=1))
    end 

    # Return the values
    LL = -optimum.minimum
    # ρ, σₐ², σᵤ² = optimum.minimizer
    ρ, σₐ², σᵤ² = ρstart, optimum.minimizer...
    return (ρ, σₐ², σᵤ², LL)
end

"""Optimize wrapper for minimizing a function of σₐ², σᵤ²"""
function myoptimize2(f, g!, θ₀, θLB, θUB, method, show_trace=false)
    try
        result = minimize(
            f, g!, 
            [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [θ₀.σₐ², θ₀.σᵤ²], 
            Fminbox(METHOD_MAP[method]),
            Optim.Options(show_trace = show_trace, show_every=1, g_tol = 1e-8, time_limit = 1)
        )
    try
        result = minimize(
            f, g!, 
            [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [θ₀.σₐ², θ₀.σᵤ²], 
            Fminbox(METHOD_MAP[method]),
            Optim.Options(show_trace = show_trace, show_every=1, g_tol = 1e-8, time_limit = 1)
        )
    catch e
        if isa(e, DomainError)
            sqrt(complex(x[2], 0))
        elseif isa(e, BoundsError)
            sqrt(x)
        end
    end
    
end





"""
    dgp(ρ, σₐ², σᵤ², β, N, T; v₀, μ₀, σ₀)

    Return simulated data from the data generating process given paramters.
    @param ρ: float in [0,1], decay rate of AR1 emissions process
    @param σₐ²: float >0, SD of year-specific emissions shock 
    @param σᵤ²: float >0, SD of region-year-specific emissions shock
    @param β: 2-vector, linear and quadratic time trend parameters
    @param v₀: float, initial emissions shock of AR1 process (v_t where t=0)
    @param b₀: Nx1 array of floats, b₀ᵢ for different regions i in {1, ..., N}
        if not given, then μ₀ and σ₀ are used to pull b₀ᵢ from a random distribution
    @param μ₀: float, mean of region-specific fixed effect distribution (b₀ᵢ)
    @param σ₀: float >0, SD of region-specific fixed effect distribution (b₀ᵢ)
"""
function dgp(ρ, σₐ², σᵤ², β, N, T;
    v₀ = 0.0, b₀ = nothing, μ₀ = 0, σ₀ = 10, random_seed::Integer = 1234)
    @info "dgp()" ρ σₐ² σᵤ² β N T b₀ μ₀ σ₀ random_seed
    # Initial conditions
    Random.seed!(random_seed)
    b = 1  # unidentified scale parameter
    # Start the emissions 200 years back when they were very small and take the last T years
    T_final = T
    T = T + 200

    # Get region-specific fixed effects if not given
    if b₀ === nothing
        b₀ = rand(Distributions.Normal(μ₀, σ₀), N)
    end

    # Random shocks
    αₜ = rand(Distributions.Normal(0, σₐ²^0.5), T)
    μᵢₜ = rand(Distributions.Normal(0, σᵤ²^0.5), N * T)
    println("True DGP values: σₐ²: ", σₐ², "   σᵤ²: ", σᵤ²)
    # assume μᵢₜ is stacked region first:
    # i.e. (i,t)=(1,1) (i,t)=(2,1) ... (i,t)=(N-1,T)  (i,t)=(N,T)

    # Fill in the aggregate shocks
    vₜ = [v₀]
    vᵢₜ = []
    for t in 1:T
        s = 0  # this period's sum of shocks
        for i in 1:N
            # Note that vₜ index starts at 1 instead of 0, so it's ahead of the others
            append!(vᵢₜ, ρ * vₜ[t] + αₜ[t] + μᵢₜ[(t-1)*N+i])
            s += last(vᵢₜ)
        end
        append!(vₜ, s / N)  # Aggregate shock = average of this period's shocks
    end
    
    data = DataFrame(t = repeat(1:T_final, inner = N),
        i = categorical(repeat(1:N, outer = T_final)),
        b₀ᵢ = repeat(b₀, outer = T_final),
        αₜ = repeat(last(αₜ,T_final), inner = N),
        μᵢₜ = last(μᵢₜ, N * T_final),
        vᵢₜ = last(vᵢₜ, N * T_final),
        vₜ = repeat(last(vₜ, T_final), inner = N))

    # Generate the resulting emissions
    data.eᵢₜ = (1 / b) * (data.b₀ᵢ + β[1]*data.t + β[2]*data.t .^ 2 + data.vᵢₜ)
    return (data)
end;
function dgp(θ, N, T, seed)
    # Helper function for estimation testing on no-trend simulations
    β = [0, 0]
    b₀ = repeat([0], N)
    return dgp(θ.ρ, θ.σₐ², θ.σᵤ², β, N, T, b₀ = b₀, random_seed=seed)
end








function estimate_rho()
    filepath = "ts_allYears_nation.1751_2014.csv"
    # See r-scripts/reproducing_andy.R result1 and result2
end

#######################################################################
#           Helper functions
#######################################################################

create_dataframe() = DataFrame(
    # This function determines the order of the columns
    seed = [], type = [], N = [], T = [],
    ρ_estimate = [], ρ_start = [], ρ_lower = [], ρ_upper = [],
    σₐ²_estimate = [],  σₐ²_start = [], σₐ²_lower = [], σₐ²_upper = [],
    σᵤ²_estimate = [],  σᵤ²_start = [],  σᵤ²_lower = [], σᵤ²_upper = [], 
    iterations = [], LL = [],
    V_norm_finalstep = [], V_norm_true=[], V = [],
    b₀₁ = [], b₀₂ = [], b₀₃ = [], b₀₄ = [], 
    β₁ = [], β₂ = [], method = [],
    runtime = [], notes=[]
)



"""If missing, return missing n-vector.
    Otherwise, return rounded var vector with missing appended to end to 
    fill out to n-vector.
"""
function optional_round(var, n, digits)
    if ismissing(var)
        if n == 1
            return missing
        else
            return repeat([missing], n)
        end
    else
        var = round.(var, digits=digits)
        if length(var) < n
            return vcat(var, repeat([missing], n - length(var)))
        end
        return var
    end
end


function add_row!(df, seed, type, N, T;
    b₀ᵢ = missing, β = missing,
    ρ=missing, σₐ²=missing, σᵤ²=missing, iterations=missing,
    LL=missing, V_norm_finalstep=missing, V_norm_true=missing, V=missing,
    params_start = missing, params_lower = missing, params_upper = missing,
    runtime=missing, notes=missing, digits=3, method = missing)

    if N == 2; b₀₁, b₀₂ = optional_round(b₀ᵢ, N, digits); b₀₃, b₀₄ = missing, missing;
    elseif N == 3; b₀₁, b₀₂, b₀₃ = optional_round(b₀ᵢ, N, digits); b₀₃ = missing;
    elseif N == 4; b₀₁, b₀₂, b₀₃, b₀₄ = optional_round(b₀ᵢ, N, digits);
    end
    β₁, β₂ = optional_round(β, 2, digits)

    ρ = optional_round(ρ, 1, 4)
    println(ρ)
    σₐ² = optional_round(σₐ², 1, digits+4)
    σᵤ² = optional_round(σᵤ², 1, digits+4)

    LL = optional_round(LL, 1, 4)
    V_norm_finalstep = optional_round(V_norm_finalstep, 1, digits)
    V_norm_true = optional_round(V_norm_true, 1, digits)

    ρ_start, σₐ²_start, σᵤ²_start = optional_round(params_start, 3, 4)
    ρ_lower, σₐ²_lower, σᵤ²_lower = optional_round(params_lower, 3, 4)
    ρ_upper, σₐ²_upper, σᵤ²_upper = optional_round(params_upper, 3, 4)

    push!(df, Dict(
    :seed=>seed, :type=>type, :N=>N, :T=>T,
    :b₀₁=>b₀₁, :b₀₂=>b₀₂, :b₀₃=>b₀₃, :b₀₄=>b₀₄,
    :β₁=>β₁,   :β₂=>β₂,
    :ρ_estimate=>ρ,    :σₐ²_estimate=>σₐ²,    :σᵤ²_estimate=>σᵤ², :LL=>LL, :iterations=>iterations,
    :V_norm_finalstep=>V_norm_finalstep,      :V_norm_true=>V_norm_true, :V=>V,
    :ρ_start=>ρ_start, :σₐ²_start=>σₐ²_start, :σᵤ²_start=>σᵤ²_start, 
    :ρ_lower=>ρ_lower, :σₐ²_lower=>σₐ²_lower, :σᵤ²_lower=>σᵤ²_lower, 
    :ρ_upper=>ρ_upper, :σₐ²_upper=>σₐ²_upper, :σᵤ²_upper=>σᵤ²_upper, 
    :method=>method,   :runtime=>runtime,     :notes=>notes
    ))
end


create_simulation_results_df() = DataFrame(
    # This function determines the order of the columns
    seed = Int64[], N = Int64[], T = Int64[],
    ρ_true = Float64[],   ρ_est = Float64[],    ρ_start = Float64[],    ρ_lower = Float64[],   ρ_upper = Float64[],
    σₐ²_true = Float64[], σₐ²_est = Float64[],  σₐ²_start = Float64[],  σₐ²_lower = Float64[], σₐ²_upper = Float64[],
    σᵤ²_true = Float64[], σᵤ²_est = Float64[],  σᵤ²_start = Float64[],  σᵤ²_lower = Float64[], σᵤ²_upper = Float64[],
    LL = Float64[], method=[], runtime = [], notes=[]
)

function add_row_simulation_results!(df, seed, type, N, T;
        θhat = missing, θ₀ = missing, θLB = missing, θUB = missing,
        LL=missing, runtime=missing, notes=missing, method = missing)
    push!(df, (
        seed=seed, type=type, N=N, T=T,
        ρ_estimate=θhat.ρ,    σₐ²_estimate=θhat.σₐ²,    σᵤ²_estimate=θhat.σᵤ², LL=LL,
        ρ_start=θ₀.ρ, σₐ²_start=θ₀.σₐ², σᵤ²_start=θ₀.σᵤ², 
        ρ_lower=θLB.ρ, σₐ²_lower=θLB.σₐ², σᵤ²_lower=θLB.σᵤ², 
        ρ_upper=θUB.ρ, σₐ²_upper=θUB.σₐ², σᵤ²_upper=θUB.σᵤ², 
        method=method,   runtime=runtime,     notes=notes
    ))
end



function write_simulation_df(df; N=2)
    # Save dataframe
    filepath = "../../data/temp/simulation_results_log(N=$N).csv"
    if isfile(filepath)
        CSV.write(filepath, df,append=true)
    else
        CSV.write(filepath, df)
    end
end


function write_estimation_df(df; N=2)
    # Round dataframe columns
    for n in names(df)
        if eltype(df[!,n]) == Float64 || eltype(df[!,n]) == Float32
            df[!,n] = round.(df[!,n], digits=3)
        end
    end
    # Save dataframe
    filepath = "../../data/temp/realdata_estimation_results_log.csv"
    if isfile(filepath)
        CSV.write(filepath, df, append=true)
    else
        CSV.write(filepath, df)
    end
end


function write_estimation_df_notrend(df, search_start; suffix="")
    # Save dataframe
    filepath = "../../data/temp/simulation_estimation$(suffix)_results_log_start$search_start.csv"
    if isfile(filepath)
        CSV.write(filepath, df, append=true)
    else
        CSV.write(filepath, df)
    end
end

function read_estimation_df_notrend(search_start)
    filepath = "../../data/temp/simulation_estimation_results_log_start$search_start.csv"
    df = CSV.read(filepath, DataFrame)
end


function write_data_df(df, data_type; N=4)
    @transform!(df, type = data_type)
    @transform!(df, N = N)
    # Save dataframe
    filepath = "../../data/temp/realdata_estimation_data.csv"
    if isfile(filepath)
        CSV.write(filepath, df, append=true)
    else
        CSV.write(filepath, df)
    end
end


function read_data(N, T)
    filepath = "../../data/sunny/clean/grouped_allYears_nation.1751_2014.csv"
    df = @chain CSV.read(filepath, DataFrame) begin
        @rsubset :Year >= 1945 && :Year < 1945+T && :group ∈ ["USA","EU","BRIC","Other"][1:N]
        @transform(:group = get.(Ref(country2num), :group, missing))
        @select(:t = :time .- 44,
                :i = categorical(:group),
                :eᵢₜ = :CO2)
        @orderby(:t, :i)
    end

    return(df)
end


function read_global_data(T)
    filepath = "../../data/sunny/clean/ts_allYears_nation.1751_2014.csv"
    df = @chain CSV.read(filepath, DataFrame) begin
        @rsubset :Year >= 1945 && :Year < 1945+T && :group ∈ ["USA","EU","BRIC","Other"][1:N]
        @transform(:group = get.(Ref(country2num), :group, missing))
        @select(:t = :time .- 44,
                :i = categorical(:group),
                :eᵢₜ = :ebar)
        @orderby(:t, :i)
    end

    return(df)
end


function get_covar_names(data)
    ilevels = unique(data.i); len = length(ilevels);
    inames = repeat(["i"], len-1) .* string.(ilevels[2:len])
    varnames = vcat(["intercept"], inames, ["t", "t^2"])
    return(varnames)
end


function get_sample_variances(data)
    σₐ² = sum((data.αₜ .- mean(data.αₜ)) .^ 2) / 2 / (T - 1)
    σᵤ² = sum((data.μᵢₜ .- mean(data.μᵢₜ)) .^ 2) / (N * T - 1)
    return(σₐ², σᵤ²)
end


function save_reformatted_data()
    N=2; T=60;
    data = read_data(N, T)
    filepath = "../../data/temp/realdata_reformated(N=$N).csv"
    CSV.write(filepath, data)
end


function translate_gls(gls, N)
    β = gls[:β][N+1:N+2]
    b₀ᵢ = gls[:β][1:N]
    for i ∈ 2:N
        b₀ᵢ[i] = gls[:β][1] + gls[:β][i]
    end
    return([β, b₀ᵢ])
end


function save_ols!(df::DataFrame, gls::Dict, N, T; data_type = "real data")
    β, b₀ᵢ = translate_gls(gls, N)
    note = "This is the first-pass OLS estimate of b₀ᵢ and β using real data (using GLS with the identity matrix)."
    add_row!(df, data_type, "First-pass OLS Estimate", N, T;
        b₀ᵢ=b₀ᵢ, β=β, iterations=0,
        runtime=Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
        notes=note
    )
end


function save_simulated_params!(df::DataFrame, params, N, T; data_type = "simulated data")
    β = params[:β]
    b₀ᵢ = params[:b₀ᵢ]
    note = "b₀ᵢ and β used to generate simulated data"
    add_row!(df, data_type, "Parameters in simulation", N, T;
        b₀ᵢ=b₀ᵢ, β=β, iterations=0,
        runtime=Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
        notes=note
    )
end


param(plist) = Dict(
    :ρ => plist[1],
    :σₐ² => plist[2],
    :σᵤ² => plist[3],
)


#######################################################################
#           Plotting
#######################################################################
# Moved to plotting_LL.jl

#######################################################################
#           Iterate to convergence!
#######################################################################
"""
Examine convergence behavior of covariance matrix by creating fake data
using known parameter values. These values may or may not be close to
any realistic values, but this allows us to examine how relative sizes
of parameters seem to effect the estimation process.
"""
function estimate_sigmas(N, T, starting_params;
        n_seeds = 10, iteration_max = 500, convergence_threshold = 1e-16,
        analytical = true)
    println()
    display([N,T,starting_params])
    @info "estimate_sigmas()" N T starting_params n_seeds iteration_max convergence_threshold
    df = create_dataframe()
    # Generate different data to see how convergence behaves
    println("\n", "_"^n_seeds)
    for seed in 1:n_seeds
        print("*")
        @info "seed: " seed
        Random.seed!(seed)
        # Starting parameter values
        ρ = starting_params[:ρ];    β = starting_params[:β]
        σₐ² = starting_params[:σₐ²];  σᵤ² = starting_params[:σᵤ²]
        LL = -Inf
        # Simuated data
        data = dgp(ρ, σₐ², σᵤ², β, N, T; random_seed = seed)
        # True values from DGP
        b₀ᵢ = unique(data.b₀ᵢ)
        V_true = Σ(ρ, σₐ², σᵤ², N, T)
        add_row!(df, seed, "True", N, T; b₀ᵢ=b₀ᵢ[:], β=β, ρ=ρ, σₐ²=σₐ², σᵤ²=σᵤ², LL=LL)
        # Observed sample variance from DGP
        σ_observed = get_sample_variances(data)
        add_row!(df, seed, "Sample", N, T; σₐ²=σ_observed[1], σᵤ²=σ_observed[2])
        # Initialize the variance matrix (identity matrix = OLS in first iteration)
        V = Diagonal(ones(length(data.eᵢₜ)))
        # Linear GLS formula
        linear_formula = @formula(eᵢₜ ~ 1 + i + t + t^2)

        Vnorm = Inf; i = 1;
        while Vnorm > convergence_threshold && i <= iteration_max
            # GLS to get residuals v
            global gls = mygls(linear_formula, data, V, N)
            v = vec(gls[:resid])
            # Uncomment the below to remove detrending/demeaning step
            # v = vec(data.vᵢₜ)

            # ML to get parameter estimates ρ, σₐ², σᵤ²
            ρ, σₐ², σᵤ², LL = mymle_3(ρ, σₐ², σᵤ², v, N, T; analytical)
            Vold = V
            V = Σ(ρ, σₐ², σᵤ², N, T)
            Vnorm = norm(Vold - V)
            i += 1
        end
        
        β = gls[:β][3:4]; b₀ᵢ[1] = gls[:β][1]; b₀ᵢ[2] = sum(gls[:β][1:2]);
        Vnorm_true = norm(V_true - V)
        add_row!(df, seed, "Estimated (calc grad, bounds)", N, T;
            b₀ᵢ=b₀ᵢ, β=β, ρ=ρ, σₐ²=σₐ², σᵤ²=σᵤ², LL=LL, iterations=i-1,
            V_norm_finalstep=Vnorm, V_norm_true=Vnorm_true,
            runtime=Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
        )
    end

    # Save dataframe of simulation results
    # write_simulation_df(df; N=N)
end




"""
Roughly estimating the model parameters to put into DGP for testing.
Also used to do final estimation...
@param starting_params: dict with ρ, σₐ², σᵤ² starting values
"""
function estimate_dgp_params(N, T, starting_params;
        iteration_max = 500, convergence_threshold = 1e-4,
        params_lower_bound = [1e-4, 1e-4, 1e-4],
        params_upper_bound = [1 - 1e-4, Inf, Inf],
        data = missing, print_results = false, data_type = "real data",
        analytical = true, method = "gradient decent", sim_params=missing)
    println("Estimating parameters of $data_type using iteration method.")
    println(N, " ", T, " ", starting_params)
    @info "estimate_dgp_params()" N T starting_params iteration_max convergence_threshold params_lower_bound params_upper_bound print_results data_type analytical method
    # dataframe to save to CSV
    df = create_dataframe()
    df2 = DataFrame(σₐ²=[],  σᵤ²=[], LL=[], grad_a=[], grad_u=[])
    # Load data, or use provided simulated data
    if ismissing(data)
        data = read_data(N, T)
    end
    if data_type == "simulated data"
        save_simulated_params!(df, sim_params, N, T)
    end
    # Initialize the variance matrix (identity matrix = OLS in first iteration)
    V = Diagonal(ones(length(data.eᵢₜ)))
    # Linear GLS formula
    linear_formula = @formula(eᵢₜ ~ 1 + i + t + t^2)
    v = zeros(N*T)

    # Starting parameter values
    ρ = starting_params[:ρ]; σₐ² = starting_params[:σₐ²];  σᵤ² = starting_params[:σᵤ²]
    Vnorm = Inf; i = 1; LL = -Inf
    while Vnorm > convergence_threshold && i <= iteration_max
        println("\nStarting iteration ", i)
        @info "Starting iteration $i"
        # GLS to estimate b₀ᵢ and β; get residuals v
        global gls = mygls(linear_formula, data, V, N)
        v = vec(gls[:resid])
        if i == 1
            save_ols!(df, gls, N, T; data_type)
        end

        # ML to get parameter estimates ρ, σₐ², σᵤ²
        ρ, σₐ², σᵤ², LL = mymle_2(ρ, σₐ², σᵤ², v, N, T;
                                lower=params_lower_bound, upper=params_upper_bound,
                                analytical=analytical, method=method)
        
        Vold = V
        V = Σ(ρ, σₐ², σᵤ², N, T)
        # Save LL and grad values to dataframe
        gradvec = nLL_grad2([0.,0.], [σₐ², σᵤ²], ρ, v, N, T)
        append!(df2, DataFrame(σₐ²=σₐ²,  σᵤ²=σᵤ², LL=LL, grad_a=[gradvec[1]], grad_u=[gradvec[2]]))
        # Convergence criteria for next loop
        Vnorm = norm(Vold - V)
        println("Vnorm: ", Vnorm)
        println("LL: ", LL, "  ρ: ", ρ, "  σₐ²: ", σₐ², "  σᵤ²: ", σᵤ²)
        i += 1
    end
    
    β, b₀ᵢ = translate_gls(gls, N)
    note = "Estimated parameters after convergence of the iterative method"
    add_row!(df, data_type, "Estimated", N, T;
        b₀ᵢ=b₀ᵢ, β=β, ρ=ρ, σₐ²=σₐ², σᵤ²=σᵤ², LL=LL, iterations=i-1,
        V_norm_finalstep=Vnorm,
        params_start = [starting_params[:ρ], starting_params[:σₐ²], starting_params[:σᵤ²]],
        params_lower = params_lower_bound, params_upper = params_upper_bound,
        runtime=Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
        notes=note
    )
    add_row!(df, "", "", N, T; notes="Row intentionally left blank for visual reasons.")

    # Save dataframe of results
    write_estimation_df(df; N=N)
    write_data_df(df2, data_type; N=N)
    println("DONE estimating parameters of $data_type.")


    # print results
    if print_results
        println(df)
    end
    return v
end




"""
Estimate model with no trend or fixed effects (just the MLE with ρ and σ's).
Used for testing estimation properties on simulation data.
θ = [ρ, σₐ², σᵤ²] true parameters for simulated data
θ₀ = starting value for parameter search
seed = random seed used for simulated data
"""
function estimate_simulation_params_notrend(N, T, θ, θ₀, seed, df;
        θLB = (ρ=0.878, σₐ²=1e-4, σᵤ²=1e-4),
        θUB = (ρ=0.879, σₐ²=Inf, σᵤ²=Inf),
        print_results = false, data_type = "simulated data",
        analytical = true, method = "gradient decent")
    println("\nEstimating parameters of $data_type using MLE")
    println(N, " ", T, " ", ", Starting search at: ", θ₀)
    @info "estimate_dgp_params()" N T θ θ₀ seed θLB θUB print_results data_type analytical method
    # Generate Simulated data
    data = dgp(θ, N, T, seed)

    # ML to get parameter estimates ρ, σₐ², σᵤ²
    # mymle2 treats first argument as fixed ρ value
    ρ, σₐ², σᵤ², LL = mymle_2(θ₀.ρ, θ₀.σₐ², θ₀.σᵤ², data.vᵢₜ, N, T;
                            θLB, θUB,
                            analytical=analytical, method=method)
    println("LL: ", LL, "  ρ: ", ρ, "  σₐ²: ", σₐ², "  σᵤ²: ", σᵤ²)
    θhat = (ρ=ρ, σₐ²=σₐ², σᵤ²=σᵤ²)

    
    note = "Estimated parameters after simple MLE with no GLS (no trend)"
    push!(df, (
        seed = seed, N = N, T = T,
        ρ_true   = θ.ρ,     ρ_est = θhat.ρ,      ρ_start = θ₀.ρ,      ρ_lower = θLB.ρ,     ρ_upper = θUB.ρ,
        σₐ²_true = θ.σₐ², σₐ²_est = θhat.σₐ²,  σₐ²_start = θ₀.σₐ²,  σₐ²_lower = θLB.σₐ², σₐ²_upper = θUB.σₐ²,
        σᵤ²_true = θ.σᵤ², σᵤ²_est = θhat.σᵤ²,  σᵤ²_start = θ₀.σᵤ²,  σᵤ²_lower = θLB.σᵤ², σᵤ²_upper = θUB.σᵤ²,
        LL = LL, method = method,
        runtime = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), notes = note
    ))

    # print results
    if print_results
        println(df)
    end
end


function get_results_σstarts(date)
    filepath = "../../data/temp/realdata_estimation_results_log_$date.csv"
    df = @chain CSV.read(filepath, DataFrame,
            select=["type", "V_norm_finalstep", "??\xb2_start", "??\xb2_start_1"],
            footerskip=184) begin
        @rsubset :V_norm_finalstep == 0
        @select(αstart=cols(Symbol("??\xb2_start")),
                μstart=cols(Symbol("??\xb2_start_1")))
    end
    return(df)
end


#######################################################################
#           Testing convergence properties of different dimensions
#######################################################################
N = 2;
T = 3;
starting_params = Dict(:ρ => 0.85,
    :σₐ² => 10,
    :σᵤ² => 10,
    :β => [1, 0.1]
)



# save_reformatted_data()

function test_dgp_params_from_real(test_no_trends=false)
    N = 4; T = 60  # 1945-2005
    # parameters estimated from data
    if N == 2
        b₀ = [200_000., 560_000.]  # from test_starting_real_estimation()
        β = [22_000., -100.]      # from test_starting_real_estimation()
    elseif N == 4
        # b₀ = [378_017.714, -45_391.353, 347_855.764, 314_643.98]
        # β = [26_518., -6.6]
        # β = [32_574.7, -108.555]
        b₀ = [3.146, -0.454, 3.78, 3.479]  # from test_starting_real_estimation() for n=4, after changing emissions units to 10,000 tons
        β = [0.265, 0] # from test_starting_real_estimation() for n=4
    end

    if test_no_trends && N == 4
        b₀ = [0, 0, 0, 0]
        β = [0, 0]
    end
    ρ = 0.8785                        # from r-scripts/reproducting_andy.R  -> line 70 result2
    # σₐ² = 1.035  # 40               # from test_starting_real_estimation()
    # σᵤ² = 16.224  # 265              # from test_starting_real_estimation()
    # σₐ² = 10^10.56  # 40               # from comparing simulated to real data
    # σᵤ² = 10^10.56   # 265              # from comparing simulated to real data
    σₐ² = 3.6288344  # from test_starting_real_estimation()
    σᵤ² = 3.6288344  # from test_starting_real_estimation()
    params = Dict(:b₀ => b₀, :β => β, :ρ => ρ, :σₐ² => σₐ², :σᵤ² => σᵤ²)
    @info "test_dgp_params_from_real()" params

    # Generate simulated data from realistic parameter estimates
    simulated_data = dgp(ρ, σₐ², σᵤ², β, N, T; v₀ = 0.0, b₀ = b₀)
    println("ρ = $ρ,  σₐ² = $σₐ²,  σᵤ² = $σᵤ²,  β = $β,  N = $N,  T = $T,  b₀ = $b₀")
    filepath = "../../data/temp/simulated_data N$N T$T rho$ρ sig_a$σₐ² sig_u$σᵤ².csv"
    CSV.write(filepath, simulated_data)
    # @df simulated_data plot(:t, :eᵢₜ, group = :i)

    # See how close the estimated parameters are to the parameters used to generate the data
    # Test various starting parameter values
    # Lock in ρ around 0.87
    # Starting parameters for the search ρ, σₐ², σᵤ²
    plist1 = [0.8785, 1, 1]
    plist2 = [0.8785, 0.1, 0.1]
    plist3 = [0.8785, 10, 10]
    plist4 = [0.8785, 100, 100]
    plist5 = [0.8785, 1000, 1000]
    plist10 = [ρ, σₐ², σᵤ²]

    lower_bounds1 = [0.878, 1e-4, 1e-4]
    upper_bounds1 = [0.879, Inf, Inf]
    lower_bounds2 = [0.05, 1e-4, 1e-4]
    upper_bounds2 = [0.99, Inf, Inf]
 
    # @time estimate_dgp_params(N, T, param(plist1); data=simulated_data,
    #     params_lower_bound=lower_bounds1,
    #     params_upper_bound=upper_bounds1)

    # @time estimate_dgp_params(N, T, param(plist2); data=simulated_data,
    #     params_lower_bound=lower_bounds1,
    #     params_upper_bound=upper_bounds1)

    # Estimate just σₐ², σᵤ²; setting ρ = ρstart
    sim_params = Dict(:β => β, :b₀ᵢ => b₀)
    @time estimate_dgp_params(N, T, param(plist10); data=simulated_data,
        params_lower_bound=lower_bounds1,
        params_upper_bound=upper_bounds1,
        print_results = false, data_type = "simulated data",
        analytical=true, sim_params)

    # @time estimate_dgp_params(N, T, param(plist4); data=simulated_data,
    #     params_lower_bound=lower_bounds1,
    #     params_upper_bound=upper_bounds1)

    # @time estimate_dgp_params(N, T, param(plist5); data=simulated_data,
    #     params_lower_bound=lower_bounds1,
    #     params_upper_bound=upper_bounds1)

end
# test_dgp_params_from_real()
# 2022-09-13 Takes about 1.5 hours to run and drives sigma_a to 0
# 2022-10-09
# 2022-12-15



"""
Estimates σ's over a range of simulated data generated from a range of σ values.
First, this creates a range of "true" σₐ² and σᵤ² values. For each pair of "true"
σₐ² and σᵤ² values, it generates 100 simulated datasets. Each simulated dataset is
generated with no trends (b₀, β = 0). Then it uses the ML method to estimate the 
σ's for each dataset to see average bias in the N=4, T=60 setting.

Nsteps = # of "true" σ² values in array to test (so full set of pairs is Nsteps × Nsteps)
Nsim = # of simulated datasets to create for each "true" σ² value pair
"""
function test_simulated_data_with_no_trends(; Nsteps = 4, Nsim = 100, search_start = 0.1)
    N = 4; T = 60  # 1945-2005
    ρ = 0.8785                        # from r-scripts/reproducting_andy.R  -> line 70 result2
    σ²base = 3.6288344  # σᵤ² from test_starting_real_estimation() after rescaling the data to units of 10,000 tons
    σ²range = range(1e-4, 2*σ²base, length=Nsteps)

    # Define a short analysis function that just takes the data
    θ₀ = (ρ=ρ, σₐ²=search_start, σᵤ²=search_start)
    θLB = (ρ=0.878, σₐ²=1e-4, σᵤ²=1e-4)
    θUB = (ρ=0.879, σₐ²=Inf, σᵤ²=Inf)

    # For each pair of "true" σₐ², σᵤ² values
    total = length(σ²range)^2; i = 0
    for σₐ² in σ²range, σᵤ² in σ²range
        result_df = create_simulation_results_df()
        θ = (ρ=ρ, σₐ²=σₐ², σᵤ²=σᵤ²)
        # Simulate and estimate Nsim times for these σ values
        println("\n\n(σₐ²=$σₐ², σᵤ²=$σᵤ²) Running $Nsim simulations $(round(i/total*100))%")
        Threads.@threads for seed in 1:Nsim
            
            estimate_simulation_params_notrend(N, T, θ, θ₀, seed, result_df;
                θLB, θUB,
                print_results = false, data_type = "simulated data",
                analytical = true, method = "gradient decent")
        end
        # Save dataframe of results
        write_estimation_df_notrend(result_df, search_start)
        # This saves the estimation results to data/temp/simulation_estimation_results_log.csv
        i += 1
    end
end

"""
Estimates σ's using different optim.jl algos to test which has least bias when 
the starting value is "far away". From previous testing in test_simulated_data_with_no_trends(),
I found the starting the search at both σ² = σ²base = 3.6288344 when the true σ² were
2*σ²base resulted in significant bias when using gradient decent. So this seems like a good case
to test gradient decent against other algos. Might even need to test a manual multi-start
algo that optimizes from multiple starting values.

- Create simulated data from both σ² = σ²base, treating trends and fixed effects as zero (b₀, β = 0).
- This eliminates the GLS step (thus no iteration is needed).
- Just testing if we can accurately find the maximizing MLE.
- Simulate and repeat estimation Nsim times, testing for each optim.jl algo

Question: Does the true σ² values maximize ML?

Nsteps = # of "true" σ² values in array to test (so full set of pairs is Nsteps × Nsteps)
Nsim = # of simulated datasets to create for each "true" σ² value pair
search_start = value of both σ²'s to start optimization search at
"""
function test_simulated_data_optim_algo(; Nsteps = 4, Nsim = 100, search_start = 0.1)
    # Nsteps = 2; Nsim = 1  # temp
    N = 4; T = 60  # 1945-2005
    ρ = 0.8785                        # from r-scripts/reproducting_andy.R  -> line 70 result2
    σ²base = 3.6288344  # σᵤ² from test_starting_real_estimation() after rescaling the data to units of 10,000 tons

    # Define a short analysis function that just takes the data
    θ = (ρ=ρ, σₐ²=2*σ²base, σᵤ²=2*σ²base)
    θ₀ = (ρ=ρ, σₐ²=search_start, σᵤ²=search_start)
    θLB = (ρ=0.878, σₐ²=1e-4, σᵤ²=1e-4)
    θUB = (ρ=0.879, σₐ²=Inf, σᵤ²=Inf)

    # For each pair of "true" σₐ², σᵤ² values
    methods = ["LBFGS", "conjugate gradient", "gradient decent", "BFGS", "momentum gradient", "accelerated gradient"]
    method = "momentum gradient"
    total = length(methods); i = 0
    for method in methods
        result_df = create_simulation_results_df()
        # Simulate and estimate Nsim times for these σ values
        @info "\n\n(σₐ²=$(θ.σₐ²), σᵤ²=$(θ.σᵤ²)) Running $Nsim simulations - $(round(i/total*100))% of outer loop"
        Threads.@threads for seed in 1:Nsim
            
            estimate_simulation_params_notrend(N, T, θ, θ₀, seed, result_df;
                θLB, θUB,
                print_results = false, data_type = "simulated data",
                analytical = true, method = method)
        end
        # Save dataframe of results
        write_estimation_df_notrend(result_df, search_start, suffix="_methods")
        # This saves the estimation results to data/temp/simulation_estimation_methods_results_log.csv
        i += 1
    end
end

σ²base = 3.6288344  
# Ran the below lines on the department server, set with 8 cores in settings.json "julia.NumThreads": "8"
# @time test_simulated_data_with_no_trends(Nsteps = 10, Nsim = 100, search_start = 3.6288344)
# @time test_simulated_data_with_no_trends(Nsteps = 10, Nsim = 100, search_start = 0.1)
# @time test_simulated_data_with_no_trends(Nsteps = 10, Nsim = 100, search_start = 10)

# Ran the below lines to test which optim.jl algos perform the best
@time test_simulated_data_optim_algo(Nsim = 1, search_start = σ²base)


function stats_simulated_estimates_with_no_trends(search_start)
    # Load estimation results from simulations
    df = @chain read_estimation_df_notrend(search_start) begin
        groupby([:σₐ²_true, :σᵤ²_true])
        @combine(
            :MAbias_a = mean(abs.(:σₐ²_est - :σₐ²_true)),
            :MAbias_u = mean(abs.(:σᵤ²_est - :σᵤ²_true)),
            :Mbias_a = mean(:σₐ²_est - :σₐ²_true),
            :Mbias_u = mean(:σᵤ²_est - :σᵤ²_true),
            :MAbias_perc_a = mean(abs.((:σₐ²_est - :σₐ²_true)./:σₐ²_true)),
            :MAbias_perc_u = mean(abs.((:σᵤ²_est - :σᵤ²_true)./:σᵤ²_true)),
            :Mbias_perc_a = mean((:σₐ²_est - :σₐ²_true)./:σₐ²_true),
            :Mbias_perc_u = mean((:σᵤ²_est - :σᵤ²_true)./:σᵤ²_true)
        )
    end

    # Plot Mean Absolute Bias
    x = unique(df.σₐ²_true)
    n = length(x)
    za = reshape(df.MAbias_a, n, n)
    zu = reshape(df.MAbias_u, n, n)
    titlea = "Mean Absolute bias in σₐ² estimation \n(search starting at white lines, 100 simulations each)\n"
    titleu = "Mean Absolute bias in σᵤ² estimation \n(search starting at white lines, 100 simulations each)\n"
    xlabel = "true σₐ²"
    ylabel = "true σᵤ²"
    p1 = heatmap_with_values_crosshairs(x, x, za, titlea, xlabel, ylabel; crosshairs=[search_start, search_start])
    p2 = heatmap_with_values_crosshairs(x, x, zu, titleu, xlabel, ylabel; crosshairs=[search_start, search_start])
    
    filepath = "../../output/simulation_plots/MLE_bias/Mean Absolute bias start $search_start.png"
    savefig(plot(p1,p2, layout=(2,1), size=(700,1000)), filepath)

    # Plot Mean Bias
    za = reshape(df.Mbias_a, n, n)
    zu = reshape(df.Mbias_u, n, n)
    titlea = "Mean bias in σₐ² estimation \n(search starting at white lines, 100 simulations each)\n"
    titleu = "Mean bias in σᵤ² estimation \n(search starting at white lines, 100 simulations each)\n"
    p1 = heatmap_with_values_crosshairs(x, x, za, titlea, xlabel, ylabel; crosshairs=[search_start, search_start])
    p2 = heatmap_with_values_crosshairs(x, x, zu, titleu, xlabel, ylabel; crosshairs=[search_start, search_start])
    
    filepath = "../../output/simulation_plots/MLE_bias/Mean bias start $search_start.png"
    savefig(plot(p1,p2, layout=(2,1), size=(700,1000)), filepath)

    # Plot Mean Absolute Bias as a percentage of the true parameter value
    za = reshape(df.MAbias_perc_a, n, n)
    zu = reshape(df.MAbias_perc_u, n, n)
    titlea = "Mean Absolute bias % in σₐ² estimation \n(search starting at white lines, 100 simulations each)\n"
    titleu = "Mean Absolute bias % in σᵤ² estimation \n(search starting at white lines, 100 simulations each)\n"
    xlabel = "true σₐ²"
    ylabel = "true σᵤ²"
    p1 = heatmap_with_values_crosshairs(x, x, za, titlea, xlabel, ylabel; crosshairs=[search_start, search_start])
    p2 = heatmap_with_values_crosshairs(x, x, zu, titleu, xlabel, ylabel; crosshairs=[search_start, search_start])
    
    filepath = "../../output/simulation_plots/MLE_bias/Mean Absolute bias percentage start $search_start.png"
    savefig(plot(p1,p2, layout=(2,1), size=(700,1000)), filepath)

    # Plot Mean Bias as a percentage of the true parameter value
    za = reshape(df.Mbias_perc_a, n, n)
    zu = reshape(df.Mbias_perc_u, n, n)
    titlea = "Mean bias % in σₐ² estimation \n(search starting at white lines, 100 simulations each)\n"
    titleu = "Mean bias % in σᵤ² estimation \n(search starting at white lines, 100 simulations each)\n"
    p1 = heatmap_with_values_crosshairs(x, x, za, titlea, xlabel, ylabel; crosshairs=[search_start, search_start])
    p2 = heatmap_with_values_crosshairs(x, x, zu, titleu, xlabel, ylabel; crosshairs=[search_start, search_start])
    
    filepath = "../../output/simulation_plots/MLE_bias/Mean bias percentage start $search_start.png"
    savefig(plot(p1,p2, layout=(2,1), size=(700,1000)), filepath)
end
# stats_simulated_estimates_with_no_trends(σ²base)
# stats_simulated_estimates_with_no_trends(0.1)
# stats_simulated_estimates_with_no_trends(10.0)

function heatmap_with_values_crosshairs(x, y, z_matrix,
        title, xlabel, ylabel; crosshairs=nothing, fontsize = 10, round_digits=1)

    heatmap(x, y, log.(abs.(z_matrix)), xlabel=xlabel, ylabel=ylabel, legend=nothing)
    title!(title)
    if !isnothing(crosshairs)
        vline!([crosshairs[1]], color=:white, label=nothing, linewidth=3)
        hline!([crosshairs[2]], color=:white, label=nothing, linewidth=3)
    end

    # Not sure the order x,y should be in -- I've only used it for x,x, so order was unimportant
    ann = [(x[i],y[j], text(round(z_matrix[j,i], digits=round_digits), fontsize, :gray, :center))
                for i in 1:length(x) for j in 1:length(y)]
    annotate!(ann, linecolor=:white)
end
# The functions below generate bias heatmaps in output\simulation_plots\MLE_bias
# stats_simulated_estimates_with_no_trends(σ²base)
# stats_simulated_estimates_with_no_trends(0.1)
# stats_simulated_estimates_with_no_trends(10.0)







function generate_boundaries(df, θLB, θUB, critical_percentile)
    # Filter data to max 10% of grid values from this recursion
    crit_LL = quantile(df.LL, critical_percentile)
    df2 = @subset(df, :LL .>= crit_LL)
    println("df $(size(df)) → df2 $(size(df2))")
    # Create new lower and upper bounds of the higher-resolution grid
    # Assuming the function is concave, this will create a grid around the max (hopefully)
    θLB_new = (σₐ² = minimum(df2.σₐ²), σᵤ² = minimum(df2.σᵤ²))
    θUB_new = (σₐ² = maximum(df2.σₐ²), σᵤ² = maximum(df2.σᵤ²))
    # Check if new boundaries are the same as old boundaries
    if θLB_new == θLB && θUB_new == θUB
        if (1-critical_percentile) <= 0.01
            println("Minimum critical percentile reached $(round(critical_percentile, digits=4)), exiting recursion.")
            return df
        end
        critical_percentile = critical_percentile * 1.1
        println("Grid boundaries unchanged... shrinking critical percentile to $critical_percentile")
        return generate_boundaries(df, θLB, θUB, critical_percentile)
    end
    return θLB_new, θUB_new
end

function recursive_LL_evaluation(f, θLB, θUB, n; df = nothing, dist_threshold=0.01, critical_percentile = 0.9)
    # Take input: grid endpoints, number of grid points, optional: dataframe
    println("θLB: $θLB"); println("θUB: $θUB")
    # Create grid of sigmas to evaluate function over
    αrange = range(θLB.σₐ², θUB.σₐ², length=n)
    μrange = range(θLB.σᵤ², θUB.σᵤ², length=n)
    # Check if distance between grid points is smaller than resolution threshold
    αdist = αrange[2] - αrange[1]
    μdist = μrange[2] - μrange[1]
    # If grid gap distance is < threshold, return dataframe
    grid_dist = max(αdist, μdist)
    if grid_dist < dist_threshold
        println("RESOLUTION REACHED: Stopping function mapping at grid distance $grid_dist")
        @info "RESOLUTION REACHED: Stopping function mapping at grid distance $grid_dist"
        return df
    end
    @info "Mapping function at grid distance $grid_dist, $((αdist, μdist))"
    println("Mapping function at grid distance $grid_dist, $((αdist, μdist))")
    # If not under threshold, evaluate in new grid and add values to df
    σpairs = [(σₐ², σᵤ²) for σₐ² in αrange for σᵤ² in μrange]
    p = Progress(n^2)
    # Create pre-defined list to store values to avoid threadding issues
    LL_list = Array{NamedTuple{}}(undef, n^2)
    Threads.@threads for i ∈ 1:n^2
        (σₐ², σᵤ²) = σpairs[i]
        LL = f(σₐ², σᵤ²)
        LL_list[i] = (σₐ²=σₐ², σᵤ²=σᵤ², LL=LL)
        next!(p)
    end
    # Create empty dataframe and fill with tuple values
    df2 = DataFrame(σₐ²=Float64[], σᵤ²=Float64[], LL=Float64[])
    for t in LL_list; push!(df2, t); end
    
    # Combine previous dataframe with current dataframe, if previous df exists
    df = isnothing(df) ? df2 : reduce(vcat, [df, df2])
    # Create new grid boundaries based on old
    θLB_new, θUB_new = generate_boundaries(df2, θLB, θUB, critical_percentile)

    # Evaluate again over new grid
    return recursive_LL_evaluation(f, θLB_new, θUB_new, n; 
        df = df, dist_threshold=dist_threshold, critical_percentile=critical_percentile)
end



"""
Find the true max of the LL function for each seeded simulated data.
Use these "true" maximums to compare to the optim.jl algos for the same seed.
"""
function find_manual_simulation_maximum(θ, seed)
    # Generate data
    # seed = 1
    N,T = 4,60
    data = dgp(θ, N, T, seed)
    v = data.vᵢₜ
    f(σₐ², σᵤ²) = -nLL(θ.ρ, σₐ², σᵤ², v, N, T)
    θLB0 = (σₐ² = 0.1, σᵤ² = 0.1)
    θUB0 = (σₐ² = 10, σᵤ² = 10)
    n = 10
    @time df = recursive_LL_evaluation(f, θLB0, θUB0, n, dist_threshold=1e-4)
    sort!(df, :LL)
    r1 = df[end,:]
    df2 = DataFrame(seed = [seed], ρ = [θ.ρ], σₐ² = [θ.σₐ²], σᵤ² = [θ.σᵤ²],
        σₐ²hat = [r1.σₐ²], σᵤ²hat = [r1.σᵤ²], LL = [r1.LL])
    return df2
end

function save_manual_simulation_maximums()
    Nsim = 100
    σ²base = 3.6288344; ρ = 0.8785
    θ = (ρ=ρ, σₐ²=σ²base, σᵤ²=σ²base)
    for seed in 1:Nsim
        df = find_manual_simulation_maximum(θ, seed)
        filepath = "../../data/temp/LL/LL_simulated_manualmax.csv"
        if isfile(filepath)
            CSV.write(filepath, df, append=true)
        else
            CSV.write(filepath, df)
        end
    end

end

save_manual_simulation_maximums()


"""
Want to plot the LL function for simulated data -- how much does it change with different seeds?
"""
function plot_simulated_LL(θ, seed)
    # SURFACE
    # using PlotlyJS
    plotly()
    # zoom in on the max
    # df from recursive_LL_evaluation()
    df2 = unique(df)
    df2 = sort(df2, :LL)
    crit_LL = quantile(df2.LL, 0.7)
    df2 = @subset(df2, :LL .> crit_LL)
    crit_LL2 = quantile(df2.LL, 0.9)
    colors = Int.(df2.LL .> crit_LL2) .+ 1
    colors = colors .+ Int.(df2.LL .== maximum(df2.LL))
    sum(Int.(df2.LL .== maximum(df2.LL)))

    gr()
    G = 1_000_000
    G = 1
    x = df2.σₐ²*G
    y = df2.σᵤ²*G
    z = df2.LL*G

    x1, y1, z1 = last.([x, y, z])

    p=Plots.plot(df2.σₐ²,  df2.σᵤ², df2.LL, zcolor = colors, m = (8, 0.8, Plots.stroke(1)), leg = false, cbar = true, w = 0,
        title = "Max Point: σₐ²=$x1, \nσᵤ²=$y1, \nLL=$z1")
    annotate!([last(x)], [last(y)], [], ["$(last(z))"])
    filepath = "../../output/simulation_plots/LL_function/LL_surface_highres_Sa$(θ.σₐ²)_Su$(θ.σᵤ²).png"
    Plots.savefig(p, filepath)



    p=PlotlyJS.plot(
        df2,
        x=:σₐ², y=:σᵤ², z=:LL,
        type="scatter3d", mode="markers",
        marker=attr(
            size=12,
            color=colors,                # set color to an array/list of desired values
            colorscale="Viridis",   # choose a colorscale
            opacity=1
        )
    )
    filepath = "../../output/simulation_plots/LL_function/LL_surface_highres_Sa$(θ.σₐ²)_Su$(θ.σᵤ²).png"
    PlotlyJS.savefig(p, filepath)


    # Save data to CSV
    df = DataFrame(σₐ² = x, σᵤ² = y, LL = z)
    filepath = "../../data/temp/LL/LL_Sa$(θ.σₐ²)_Su$(θ.σᵤ²)_seed$(seed).csv"
    if isfile(filepath)
        CSV.write(filepath, df, append=true)
    else
        CSV.write(filepath, df)
    end


    X = reshape(x, n, n)  # should be columns of x values (changing across the row - identical rows)
    Y = reshape(y, n, n)  # should be rows of y values (changing down the column - identical columns)
    Z = reshape(z, n, n)


    # SURFACE
    # using PlotlyJS
    plotly()
    # zoom in on the max
    df2 = @subset(df, :σₐ² .> 2, :σᵤ² .> 2)  # need to subset the df so that there is still a grid of points
    x = unique(df2.σₐ²)
    y = unique(df2.σᵤ²)
    Z = reshape(df2.LL, length(x), length(y))
    PlotlyJS.plot(PlotlyJS.surface(x=x, y=y, z=Z),
        Layout(;title="LogLikelihood for simulated data generated with σ²=$(σ²base)",
            xaxis=attr(title="σα²"),
            yaxis=attr(title="σμ²")))

    # CONTOUR
    PlotlyJS.plot(PlotlyJS.contour(z=Z,
        x=x, # horizontal axis
        y=y, # vertical axis
        contours_start=-341,
        contours_end=maximum(z),
        # contours_size=1,
        contours_coloring="heatmap"),
        Layout(;title="LogLikelihood for simulated data generated with σ²=$(σ²base)",
                xaxis=attr(title="σα²"),
                yaxis=attr(title="σμ²")))

    p6 = PlotlyJS.contour(z=log.(-Z),
        x=rang, # horizontal axis
        y=rang, # vertical axis
        # contours_start=5,
        # contours_end=log.(-Z)),
        # contours_size=1,
        contours_coloring="heatmap")
    p7 = PlotlyJS.scatter(;x=[], y=[], mode="markers")
    PlotlyJS.plot(p6)


    filepath = "../../output/simulation_plots/LL_function/LL_surface_Sa$(θ.σₐ²)_Su$(θ.σᵤ²).png"
    savefig(p1, filepath)




    p3 = surface(x, x, z_mat, # xscale = :log10, yscale = :log10,
        xlabel="σα²",
        ylabel="σμ²",
        zlabel="log₁₀(LL)",
        # zlims = (-3.4849, -3.48488),
        title="Log Likelihood")
    p3

    # z_mat = [f(σₐ², σᵤ²) for σₐ² in ]
    # @time plot(f_a, 0.01, 10)
    println("Plotting....")
    y = f_u.(x)
    # @time plot(x, y)
    plot(x, -y, yaxis=:log, xlabel="")
    vline!([σ²base])
    # @time plot(f_u, 0.01, 10, yaxis=:log)
    
end



function test_starting_real_estimation()
    println("Starting series of estimations using real data.")
    # Starting parameters for the search ρ, σₐ², σᵤ²
    ρ = 0.8785
    plist1 = [ρ, 1, 1]
    plist2 = [0.1, 0.1, 0.1]
    plist3 = [0.1, 10, 10]
    plist4 = [0.1, 100, 100]
    plist5 = [0.1, 1000, 1000]
    plist6 = [0.99, 0.1, 0.1]
    plist7 = [0.99, 10, 10]
    plist8 = [0.99, 100, 100]
    plist9 = [0.99, 1000, 1000]
    plist10 = [ρ, 50000^2, 50000^2]
    plist11 = [ρ, 10^10.5, 10^10.5]
    plist12 = [ρ, 1.0000000000953714e-10, 3.628834416768247]
    plist13 = [ρ, 10, 10]
    plist14 = [ρ, 0.01, 0.01]

    lower_bounds1 = [0.878, 1e-10, 1e-10]
    upper_bounds1 = [0.879, Inf, Inf]
    N = 4;
    T = 60;
    # @time estimate_dgp_params(N, T, param(plist1);
    #     params_lower_bound=lower_bounds1,
    #     params_upper_bound=upper_bounds1)
    plist = [ρ, 0.01, 3]
    v = @time estimate_dgp_params(N, T, param(plist);
            params_lower_bound=lower_bounds1,
            params_upper_bound=upper_bounds1,
            analytical=true, method = "BFGS",
            iteration_max = 100)

    date = "2022-12-15"
    austarts = get_results_σstarts(date)
    for row in eachrow(df)
        plist = [ρ, row.αstart, row.μstart]
        v = @time estimate_dgp_params(N, T, param(plist);
                params_lower_bound=lower_bounds1,
                params_upper_bound=upper_bounds1,
                analytical=true, method = "BFGS",
                iteration_max = 100)
    end

    austarts = [[25, 10], [50, 1000], [50, 0.1]]
    methods = ["LBFGS", "conjugate gradient", "gradient decent", "BFGS", "momentum gradient", "accelerated gradient"]
    for au in austarts, method in methods
        plist = [ρ, au[1], au[2]]
        v = @time estimate_dgp_params(N, T, param(plist);
                params_lower_bound=lower_bounds1,
                params_upper_bound=upper_bounds1,
                analytical=true, method = method,
                iteration_max = 100)
    end

    # vals = [0.01, 0.1, 1, 3, 6, 10, 25, 50, 100, 1000]
    # for a in vals, u in vals
    #     plist = [ρ, a, u]
    #     v = @time estimate_dgp_params(N, T, param(plist);
    #             params_lower_bound=lower_bounds1,
    #             params_upper_bound=upper_bounds1,
    #             analytical=true, method = "BFGS")
    # end

    # @time estimate_dgp_params(N, T, param(plist1);
    #     params_lower_bound=lower_bounds1,
    #     params_upper_bound=upper_bounds1,
    #     analytical=true, method = "LBFGS")

    # @time estimate_dgp_params(N, T, param(plist1);
    #     params_lower_bound=lower_bounds1,
    #     params_upper_bound=upper_bounds1,
    #     analytical=true, method = "gradient decent")

    # @time estimate_dgp_params(N, T, param(plist1);
    #     params_lower_bound=lower_bounds1,
    #     params_upper_bound=upper_bounds1,
    #     analytical=true, method = "conjugate gradient")

    # @time estimate_dgp_params(N, T, param(plist1);
    #     params_lower_bound=lower_bounds1,
    #     params_upper_bound=upper_bounds1,
    #     analytical=true, method = "momentum gradient")

    # @time estimate_dgp_params(N, T, param(plist1);
    #     params_lower_bound=lower_bounds1,
    #     params_upper_bound=upper_bounds1,
    #     analytical=true, method = "accelerated gradient")
    # results written in "/data/temp/realdata_estimation_results_log.csv"

    # @time estimate_dgp_params(N, T, param(plist2))

    # @time estimate_dgp_params(N, T, param(plist3))

    # @time estimate_dgp_params(N, T, param(plist4))

    # @time estimate_dgp_params(N, T, param(plist5))

    # @time estimate_dgp_params(N, T, param(plist6))

    # @time estimate_dgp_params(N, T, param(plist7))

    # @time estimate_dgp_params(N, T, param(plist8))

    # @time estimate_dgp_params(N, T, param(plist9))
    return v
end
# v = test_starting_real_estimation()
# plot_gradient(v)
# about 2 minutes to finish


function test_bounds_real_estimation()
    # lower bounds on params: ρ, σₐ², σᵤ²
    lower_bounds1 = [1e-4, 1e-4, 1e-4]
    lower_bounds2 = [0.1, 1, 1]
    upper_bounds1 = [1, Inf, Inf]
    upper_bounds2 = [0.85, Inf, Inf]
    N = 2;
    T = 60;
    @time estimate_dgp_params(N, T, starting_params;
        params_lower_bound=lower_bounds1,
        params_upper_bound=upper_bounds1)


    @time estimate_dgp_params(N, T, starting_params;
        params_lower_bound=lower_bounds1,
        params_upper_bound=upper_bounds2)


    @time estimate_dgp_params(N, T, starting_params;
        params_lower_bound=lower_bounds2,
        params_upper_bound=upper_bounds1)


    @time estimate_dgp_params(N, T, starting_params;
        params_lower_bound=lower_bounds2,
        params_upper_bound=upper_bounds2)
    # T=3:   53.165765 seconds (136.57 M allocations: 8.168 GiB,  4.27% gc time, 90.99% compilation time)
    # T=60:  92.230013 seconds (354.54 M allocations: 24.525 GiB, 4.01% gc time, 51.16% compilation time)
end

# Timing test between analytical gradient and calcualted gradient
# RESULT: analytical gradients take about 5/6 of the time as calculating gradients,
#       more "allocations" but less memory is written to
function time_gradient_methods()
    @time estimate_sigmas(2, 50, starting_params; n_seeds=10, analytical=true)
    # 232.415821 seconds (1.20 G allocations: 101.854 GiB, 3.80% gc time, 7.80% compilation time)
    @time estimate_sigmas(2, 50, starting_params; n_seeds=10, analytical=true)
    # 267.077041 seconds (1.15 G allocations: 99.125 GiB, 3.25% gc time)
    @time estimate_sigmas(2, 50, starting_params; n_seeds=10, analytical=false)
    # 316.247974 seconds (655.02 M allocations: 139.433 GiB, 3.79% gc time, 0.08% compilation time)
    @time estimate_sigmas(2, 50, starting_params; n_seeds=10, analytical=false)
    # 315.076701 seconds (654.42 M allocations: 139.398 GiB, 4.04% gc time)
end
# time_gradient_methods()

function test_estimation()
    # Do the estimates get closer as T gets larger? (consistency)
    println("\nTesting T")
    for T ∈ [5,10,25,50,100]
        estimate_sigmas(2, T, starting_params; n_seeds=10)
    end

    # Do the estimates get closer as N gets larger? (consistency)
    println("\nTesting N")
    for N ∈ [3,4]
        estimate_sigmas(N, 50, starting_params; n_seeds=10)
    end

    # Is the quality of the estimates sensitive to the relative size of σₐ²
    println("\nTesting σₐ²")
    for σₐ² ∈ [0.1, 1, 5, 10]
        starting_params = Dict(:ρ => 0.85,
            :σₐ² => σₐ²,
            :σᵤ² => 1,
            :β => [1, 0.1]
        )
        estimate_sigmas(2, 50, starting_params; n_seeds=10)
    end

    # Is the quality of the estimates sensitive to the relative size of σᵤ²
    println("\nTesting σᵤ²")
    for σᵤ² ∈ [0.1, 1, 5, 10]
        starting_params = Dict(:ρ => 0.85,
            :σₐ² => 1,
            :σᵤ² => σᵤ²,
            :β => [1, 0.1]
        )
        estimate_sigmas(2, 50, starting_params; n_seeds=10)
    end
end

function check_fixed_effects()
    # Define paramters with zero error variance to get smooth data
    N, T = 4, 60
    b₀ = [3.146, -0.454, 3.78, 3.479]  # from test_starting_real_estimation() for n=4, after changing emissions units to 10,000 tons
    β = [0.265, 0] # from test_starting_real_estimation() for n=4
    ρ = 0.8785                        # from r-scripts/reproducting_andy.R  -> line 70 result2
    σₐ² = 0  # from test_starting_real_estimation()
    σᵤ² = 0  # from test_starting_real_estimation()
    @info "check_fixed_effects()"     b₀ β ρ σₐ² σᵤ²

    # Generate simulated data
    simulated_data = dgp(ρ, σₐ², σᵤ², β, N, T; v₀ = 0.0, b₀ = b₀)

    # Estimate fixed effects
    plist = [ρ, σₐ², σᵤ²]
    lower_bounds = [0.878, 1e-4, 1e-4]
    linear_formula = @formula(eᵢₜ ~ 1 + i + t + t^2)
    V = Diagonal(ones(length(simulated_data.eᵢₜ)))
    gls = mygls(linear_formula, simulated_data, V, N)
    β2, b₀ᵢ = translate_gls(gls, N)

    # Predict values using estimated
    simulated_data2 = dgp(ρ, σₐ², σᵤ², β2, N, T; v₀ = 0.0, b₀ = b₀ᵢ)

    # Compare differences
    maximum(abs.(simulated_data.eᵢₜ - simulated_data2.eᵢₜ))

    # Compare plots
    @df simulated_data plot(:t, :eᵢₜ, group = :i)
    @df simulated_data2 plot!(:t, :eᵢₜ, group = :i)
end


# Close logging file
@info "End" datetime=Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
close(io)





"""
Developer Notes:
- Try simple case with rho = 1 or zero or other cases Larry mentions

Using the residuals from the true beta and FE, we get a little bit closer 
to small Vnorm between the trueV norm and the estV norm (24.2 instead of 27.4)


# Testing glm function (can't take non-diagonal matrix)
gls = glm(linear_formula, data, Normal(), IdentityLink(), wts=ones(length(data.eᵢₜ)))

# Creating a more analytical gradient
objective(ρ, σₐ², σᵤ², v) = norm(maximand(ρ, σₐ², σᵤ², v));
ρ, σₐ², σᵤ² = ρstart, σₐ²start, σᵤ²start
for k in 1:iteration_max
    grad = gradient(objective(ρ, σₐ², σᵤ², v))
# Could also reverse-mode auto-differentiate to get complete analytical gradient


# Testing symbolic gradient
@variables ρ σₐ² σᵤ²
V = Σ(2, 3; verbose=false)
Vinv = inv(V)
Vinv = substitute(Vinv, Dict(true => 1))
println(Vinv, "\n")
v = ones(N*T)
nLL(v, N, T) = (1/2)*( v' * inv(Σ(N, T)) * v + log(det(Σ(N, T))) );
nLL = nLL(v, 2, 3)
println(nLL, "\n")
nLLgrad = Symbolics.gradient(nLL, [ρ,σₐ²,σᵤ²])
println(nLLgrad)











GLS: could decompose using Cholskey and augment the data than feed it into glm 
linear model and use CovarianceMatrices from https://gragusa.org/code/ to get HAC SEs
M.diagonalize()

Iterate many times creating new data and save the vnorm from trueV, see the 
dist of vnorm and distance from true parameter values. Is it just the 
randomness of small sampling?

Also create general V function building function from N, T that returns a 
    function to generate V given params (general function should be used 
        once only for given def of N,T)

Want to compare N=2, small T to N=4, large T. Large T should have better 
convergence? Since we can get more info about variance from more obs



REFERENCES:
If you use Symbolics.jl, please cite this paper
@article{gowda2021high,
  title={High-performance symbolic-numerics via multiple dispatch},
  author={Gowda, Shashi and Ma, Yingbo and Cheli, Alessandro and Gwozdz, Maja and Shah, Viral B and Edelman, Alan and Rackauckas, Christopher},
  journal={arXiv preprint arXiv:2105.03949},
  year={2021}
}


"""

