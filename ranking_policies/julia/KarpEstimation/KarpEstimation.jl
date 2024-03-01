#==============================================================================
file: main.jl
description: Main file for project. 
    Contains: model functions, estimation algorithms, real data loading
author: Aaron C Watt (UCB Grad Student, Ag & Resource Econ)
email: aaron@acwatt.net
created: 2023-06-13
last update: 
See docs/code/main.md for more notes
Copied from covariance_matrix.jl on 2023-06-05
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
using Parameters # for @unpack

import Revise
#! Add ROOT path stuff here and make all relative paths to root
# cd("ranking_policies/julia/KarpEstimation")
Revise.includet("../reg_utils.jl")  # mygls()

#=
]add Random Distributions DataFrames DataFramesMeta CategoricalArrays LinearAlgebra StatsModels StatsPlots GLM Optim FiniteDifferences Dates Latexify Logging CSV Revise Parameters
=#

Revise.includet("Utilities/HelperFunctions.jl")  # HF
Revise.includet("Estimators/Estimators.jl")  # Est
module Model
    include("Model/Model.jl")
end

"""
NOTES:
Purpose: Estimate the parameters ρ, σₐ², σᵤ² of Larry Karp's emissions
    model using iterated Feasible GLS and MLE

How: Most of this script defines functions that represent equations from
    the model. Finally, in the "Iterate to convergence!" section at the
    end, the equations are used to iteratively estimate:
    (1) the GLS parameters b₀ᵢ and β (time trend) as inputs to (2)
    (2) the MLE parameters ρ, σₐ², σᵤ²

First, I test the estimation method on simulated data where we know the parameter values
    to see how close the estimates are to the "true" values. I test
    the estimates using different values of T, N, σₐ², σᵤ².

Lastly, I (still need to) estimate the paramters on the actual data that
    Larry provided.
Status: in the process of moving functions to separate files
Time to run: Most of the functions at the end that actually run the
    estimations also have time estimates in a comment after them. These
    were run on a linux computer with 32 GB of RAM and 
    AMD Ryzen 3 2300x quad-core processor. Times may be more or less
    depending on your system. Julia does multithreading automatically,
    meaning it will take up all available CPU during most compuations
    until the script has complete.

Last update: 2023-07-10
"""

#######################################################################
#           Setup
#######################################################################
# CODE = dirname(@__FILE__)
# ROOT = HF.get_project_root_path()

io = open("log.txt", "a")
logger = SimpleLogger(io)
# debuglogger = ConsoleLogger(stderr, Logging.Debug)
global_logger(logger)
s = "\n"^20 * "="^60 * "\nCOVARIANCE MATRIX ESTIMATION BEGIN\n" * "="^60
@info s datetime=Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
num2country = Dict(1=>"USA", 2=>"EU", 3=>"BRIC", 4=>"Other")
country2num = Dict("USA"=>1, "EU"=>2, "BRIC"=>3, "Other"=>4)



#######################################################################
#              Estimate parameters
#######################################################################


# Reformat raw data into useful CSV
# save_reformatted_data()

#! Add in MB rho estimation here
"""Estimate autocorrelation parameter ρ"""
function estimate_rho()
    filepath = "ts_allYears_nation.1751_2014.csv"
    # See r-scripts/reproducing_andy.R result1 and result2
end


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

# Timing test between analytical gradient and calcualted gradient

"""Time estimate_sigmas with analytical and numerical gradient methods.

    RESULT: analytical gradients take about 5/6 of the time as calculating gradients,
    more "allocations" but less memory is written to
"""
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

"""Test estimate_sigmas for different parameter values."""
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



"""
Roughly estimating the model parameters to put into simulated DGP for estimator testing.
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


"""Test estimate_dgp_params for many combinations of starting params"""
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

"""Test estimate_dgp_params for different parameter lower and upper bounds."""
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
    data = Model.dgp(θ, N, T, seed)

    # ML to get parameter estimates ρ, σₐ², σᵤ²
    # mymle2 treats first argument as fixed ρ value
    ρ, σₐ², σᵤ², LL = Est.MLE.mymle_2(θ₀.ρ, θ₀.σₐ², θ₀.σᵤ², data.vᵢₜ, N, T;
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



#######################################################################
#           Testing convergence properties along different dimensions
#######################################################################

"""Test the bias of the estimator for estimated values of the parameters."""
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


"""Single test of base parameter estimation on simulated data with no trends"""
function test_simulated_data_with_no_trends(; simple = true)
    N = 4; T = 60  # 1945-2005
    ρ = 0.8785  # from r-scripts/reproducting_andy.R  -> line 70 result2
    σ²base = 3.6288344  # σᵤ² from test_starting_real_estimation() after rescaling the data to units of 10,000 tons

    θ₀ = (ρ=ρ, σₐ²=σ²base/2, σᵤ²=σ²base/2)  # Starting parameter in optimization search
    θ = (ρ=ρ, σₐ²=σ²base, σᵤ²=σ²base)  # True paramter
    θLB = (ρ=0.878, σₐ²=1e-4, σᵤ²=1e-4)  # Lower bound of search
    θUB = (ρ=0.879, σₐ²=Inf, σᵤ²=Inf)  # Upper bound of search

    result_df = HF.create_simulation_results_df()
    estimate_simulation_params_notrend(N, T, θ, θ₀, seed, result_df;
        θLB, θUB,
        print_results = true, data_type = "simulated data",
        analytical = true, method = "gradient decent"
    )

    # For each pair of "true" σₐ², σᵤ² values
    total = length(σ²range)^2; i = 0
    for σₐ² in σ²range, σᵤ² in σ²range
        result_df = HF.create_simulation_results_df()
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
        HF.write_estimation_df_notrend(result_df, search_start)
        # This saves the estimation results to data/temp/simulation_estimation_results_log.csv
        i += 1
    end
end


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
    ρ = 0.8785  # from r-scripts/reproducting_andy.R  -> line 70 result2
    σ²base = 3.6288344  # σᵤ² from test_starting_real_estimation() after rescaling the data to units of 10,000 tons
    σ²range = range(1e-4, 2*σ²base, length=Nsteps)

    # Define a short analysis function that just takes the data
    θ₀ = (ρ=ρ, σₐ²=search_start, σᵤ²=search_start)
    θLB = (ρ=0.878, σₐ²=1e-4, σᵤ²=1e-4)
    θUB = (ρ=0.879, σₐ²=Inf, σᵤ²=Inf)

    # For each pair of "true" σₐ², σᵤ² values
    total = length(σ²range)^2; i = 0
    for σₐ² in σ²range, σᵤ² in σ²range
        result_df = HF.create_simulation_results_df()
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

σ²base = 3.6288344  
# Ran the below lines on the department server, set with 8 cores in settings.json "julia.NumThreads": "8"
# @time test_simulated_data_with_no_trends(Nsteps = 10, Nsim = 100, search_start = 3.6288344)
# @time test_simulated_data_with_no_trends(Nsteps = 10, Nsim = 100, search_start = 0.1)
# @time test_simulated_data_with_no_trends(Nsteps = 10, Nsim = 100, search_start = 10)

function test_optim_algo(;
    datatype="simulated", distributed=false,
    Nsim = 100, search_start = 0.1
    )
    N = 4; T = 60  # 1945-2005
    ρ = 0.8785                        # from r-scripts/reproducting_andy.R  -> line 70 result2
    σ²base = 3.6288344  # σᵤ² from test_starting_real_estimation() after rescaling the data to units of 10,000 tons

    # Define parameters for generating simulated data
    θ = (ρ=ρ, σₐ²=2*σ²base, σᵤ²=2*σ²base)
    # Define parameters for search
    search_params = (
        θ₀ = (ρ=ρ, σₐ²=search_start, σᵤ²=search_start),
        θLB = (ρ=0.878, σₐ²=1e-4, σᵤ²=1e-4),
        θUB = (ρ=0.879, σₐ²=Inf, σᵤ²=Inf)
    )

    methods = ["LBFGS", "conjugate gradient", "gradient decent", "BFGS"] #, "momentum gradient", "accelerated gradient"]
    # MomentumGradientDescent and AcceleratedGradientDescent do not work with Fminbox

    # Iterate over method-seeds pairs
    iters = [(m, n) for m in methods for n in 1:Nsim]
    println("# of tasks: $(length(iters))")
    dfs = Array{DataFrame}(undef, length(iters))


    # Summarize results over all seeds
    cat_df = reduce(vcat, dfs)
    summary_df = HF.summarize_simulation_results(cat_df)
    # Save dataframe of results
    HF.write_estimation_df_notrend(summary_df, "all", suffix="_methods_summaries")
    # This saves the estimation results to data/temp/simulation_estimation_methods_summaries_results_log.csv
    return summary_df

end

test_optim_algo(;"simulated", distributed=true, Nsim = 100, search_start = 0.1)

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
function test_simulated_data_optim_algo_parallel(iters, θ, search_params)
    @unpack θ₀, θLB, θUB = search_params
    p = Progress(length(iters))
    Threads.@threads for i in 1:length(iters)
        println("Sim $i")
        method, seed = iters[i]
        result_df = HF.create_simulation_results_df()
        # Simulate and estimate for these σ values
        estimate_simulation_params_notrend(N, T, θ, θ₀, seed, result_df;
            θLB, θUB,
            print_results = false, data_type = "simulated data",
            analytical = true, method = method)
        dfs[i] = result_df
        next!(p)
    end
    return dfs
end
"""
Distributed Estimates σ's using different optim.jl algos to test which has least bias when 
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
function test_simulated_data_optim_algo_distributed(iters, θ, search_params)
    @unpack θ₀, θLB, θUB = search_params
    @showprogress @distributed for i in 1:length(iters)
        println("Sim $i")
        method, seed = iters[i]
        result_df = HF.create_simulation_results_df()
        # Simulate and estimate for these σ values
        estimate_simulation_params_notrend(N, T, θ, θ₀, seed, result_df;
            θLB, θUB,
            print_results = false, data_type = "simulated data",
            analytical = true, method = method)
        dfs[i] = result_df
    end
    return dfs
end


# # Ran the below lines to test which optim.jl algos perform the best
# println("Comparing Optim Algos: 1 sim")
# println("# of Threads:$(Threads.nthreads())")
# df_ = @time test_simulated_data_optim_algo(Nsim = 1, search_start = σ²base)
# println("Comparing Optim Algos: 16 sim")
# println("# of Threads:$(Threads.nthreads())")
# df_ = @time test_simulated_data_optim_algo(Nsim = 16, search_start = σ²base)
# # local: 218.963896 second (7.81 G allocations: 369.491 GiB, 41.93% gc time, 0.11% compilation time: 28% of which was recompilation)
# # remote batch: 916.522670 seconds (7.85 G allocations: 371.329 GiB, 30.08% gc time)
# # remote cmd: 895.401775 seconds (7.89 G allocations: 373.220 GiB, 27.09% gc time)


# Ran the below lines to test which optim.jl algos perform the best
println("Comparing Optim Algos: 1 sim")
println("# of Threads:$(Threads.nthreads())")
df_ = @time test_simulated_data_optim_algo_distributed(Nsim = 1, search_start = σ²base)
println("Comparing Optim Algos: 16 sim")
println("# of Threads:$(Threads.nthreads())")
df_ = @time test_simulated_data_optim_algo_distributed(Nsim = 16, search_start = σ²base)
# local: 218.963896 second (7.81 G allocations: 369.491 GiB, 41.93% gc time, 0.11% compilation time: 28% of which was recompilation)
# remote batch: 916.522670 seconds (7.85 G allocations: 371.329 GiB, 30.08% gc time)
# remote cmd: 895.401775 seconds (7.89 G allocations: 373.220 GiB, 27.09% gc time)

# println("Comparing Optim Algos: 100 sim")
# println("# of Threads:$(Threads.nthreads())")
# df_ = @time test_simulated_data_optim_algo(Nsim = 100, search_start = σ²base)
# local: 1245.314972 seconds (46.14 G allocations: 2.133 TiB, 43.12% gc time, 0.02% compilation time)
# remote cmd: 975.362874 seconds (7.81 G allocations: 369.438 GiB, 19.01% gc time)

# @time test_simulated_data_optim_algo(Nsim = 1, search_start = σ²base)
# @time test_simulated_data_optim_algo(Nsim = 10, search_start = σ²base)
# 114.5 seconds to run with 0% compilation time































"""
Estimate σ's for single optim algorithm over range of N (# of time periods)
"""
function test_simulated_data_consistency(;)
    println("temp")
end




#######################################################################
#           Recursive LL maximization on grid
#######################################################################

"""Return xth-percentile paramter boundaries on log likelihood surface.
where x is the critical percentile, and the boundaries are the xth percentile of the LL values.
    This is used to create new grid boundaries for the next recursion of the LL function.
    If the new boundaries are the same as the old boundaries, the critical percentile is
    increased by 10% and the function is called again.
"""
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

"""Recursively zoom in on maximizing area of log likelihood surface."""
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

"""Find the true max of the LL function for each seeded simulated data.
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
# save_manual_simulation_maximums()





#######################################################################
#           Plots
#######################################################################
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







#######################################################################
#           Statistics of estimations
#######################################################################

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








#######################################################################
#           Shutdown
#######################################################################
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

