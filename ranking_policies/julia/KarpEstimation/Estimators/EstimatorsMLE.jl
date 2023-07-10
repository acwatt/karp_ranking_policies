"""
Functions to estimate Maximum Likelihood model
"""
module MLE
using Optim
using LinearAlgebra
include("../Model/Model.jl")  # Model


#######################################################################
#           Helper Functions
#######################################################################
"""Function to fill the NT x NT matrix with elements defined by ijs_func"""
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


#######################################################################
#           Likelihood Functions
#######################################################################
"""Negative Log Likelihood (want to minimize)"""
function nLL(ρ, σₐ², σᵤ², v, N, T)
    V = Σ(ρ,σₐ²,σᵤ²,N,T)
    nLL = (1/2) * ( v'*V^-1*v + log(det(big.(V))) )
    nLL = Float64(nLL)
    return(nLL)
end
# Define using parameter vector θ
function nLL(θ, v, N, T)
    V = Σ(θ.ρ, θ.σₐ², θ.σᵤ², N, T)
    nLL = (1/2) * ( v'*V^-1*v + log(det(big.(V))) )
    nLL = Float64(nLL)
    return(nLL)
end

"""Partial derivative of negative log-likelihood function, with respect to symbol y"""
function ∂nLL∂y(y, ρ, σₐ², σᵤ², v, N, T)
    ijs_fun(ρ, σₐ², σᵤ², i, j, s, N) = ∂Σ∂y(y, ρ, σₐ², σᵤ², i, j, s, N)
    ∂V∂y = fill_matrix(ijs_fun, ρ, σₐ², σᵤ², N, T)

    V = Σ(ρ,σₐ²,σᵤ²,N,T)
    vV⁻¹∂V∂yV⁻¹v = v'*V^-1*∂V∂y*V^-1*v
    V⁻¹∂V∂y = V^-1*∂V∂y

    ∂LL∂y = -1/2 * (-vV⁻¹∂V∂yV⁻¹v + tr(V⁻¹∂V∂y))
    return(-∂LL∂y)
end


############################### 3-d gradients ###############################
"""Analytical gradient of the negative log likelihood for θ = (ρ, σₐ², σᵤ²)"""
function nLL_grad!(gradient_vec, params, v, N, T)
    ρ = params[1]; σₐ² = params[2]; σᵤ² = params[3]
    gradient_vec[1] = ∂nLL∂y("ρ", ρ, σₐ², σᵤ², v, N, T)
    gradient_vec[2] = ∂nLL∂y("σₐ²", ρ, σₐ², σᵤ², v, N, T)
    gradient_vec[3] = ∂nLL∂y("σᵤ²", ρ, σₐ², σᵤ², v, N, T)
    return gradient_vec
end
"""Finite Difference gradient of the negative log likelihood for θ = (ρ, σₐ², σᵤ²)"""
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

############################### 2-d gradients ###############################
"""Analytical gradient of the negative log likelihood for θ = (σₐ², σᵤ²)"""
function nLL_grad2(gradient_vec, params, ρ, v, N, T)
    σₐ² = params[1]; σᵤ² = params[2]
    gradient_vec[1] = ∂nLL∂y("σₐ²", ρ, σₐ², σᵤ², v, N, T)*N
    gradient_vec[2] = ∂nLL∂y("σᵤ²", ρ, σₐ², σᵤ², v, N, T)
    return gradient_vec
end
"""Finite Difference gradient of the negative log likelihood for θ = (σₐ², σᵤ²)"""
function nLL_grad_fidiff2!(gradient_vec, σₐ²σᵤ², ρ, v, N, T)
    g = FiniteDifferences.grad(central_fdm(5, 1),
                           σₐ²σᵤ² -> nLL(ρ, σₐ²σᵤ²[1], σₐ²σᵤ²[2], v, N, T),
                           σₐ²σᵤ²)
    # update the gradient vector
    gradient_vec[1] = g[1][1]  # σₐ²
    gradient_vec[2] = g[1][2]  # σᵤ²
    return gradient_vec
end

############################### gradient tests ###############################
# println("Fin. Diff. gradient: ", nLL_grad_fidiff!([0.,0.,0.], params0, v, N, T))
# println("Fin. Diff. gradient: ", nLL_grad_fidiff2!([0.,0.], params0[2:3], ρstart, v, N, T))

# println("Fin. Diff. gradient: ", nLL_grad_fidiff!([0.,0.,0.], params0, v, N, T))
# println("Fin. Diff. gradient: ", nLL_grad_fidiff2!([0.,0.], params0[2:3], ρstart, v, N, T))


#######################################################################
#                        Optimization Functions
#######################################################################
"""Optimize wrapper for minimizing a function of σₐ², σᵤ²"""
function myoptimize2(f, g!, θ₀, θLB, θUB, method, show_trace=false)
    try
        result = minimize(
            f, g!, 
            [θLB.σₐ², θLB.σᵤ²], [θUB.σₐ², θUB.σᵤ²], [θ₀.σₐ², θ₀.σᵤ²], 
            Fminbox(METHOD_MAP[method]),
            Optim.Options(show_trace = show_trace, show_every=1, g_tol = 1e-8, time_limit = 1)
        )
    #! What was I trying to catch here?
    catch e
        if isa(e, DomainError)
            sqrt(complex(x[2], 0))
        elseif isa(e, BoundsError)
            sqrt(x)
        end
    end
    
end



#######################################################################
#                            MLE Functions
#######################################################################
"""Map of string keywords to Optim.jl optimization search algorithms."""
METHOD_MAP = Dict("gradient decent" => GradientDescent(),
                  "conjugate gradient" => ConjugateGradient(),
                  "BFGS" => BFGS(),
                  "LBFGS" => LBFGS(),
                  "momentum gradient" => MomentumGradientDescent(),
                  "accelerated gradient" => AcceleratedGradientDescent()
)

"""
    mymle(ρstart, σₐ²start, σᵤ²start, v)

Return likelihood-maximizing values of ρ, σₐ², and σᵤ², given the residuals 

Given vector of residuals, minimize the 

# Arguments
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


#######################################################################
#                            Test Functions
#######################################################################



end