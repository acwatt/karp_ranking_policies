
println(@__DIR__)
using Pkg; Pkg.activate(joinpath(@__DIR__, "turing_test_env"))
# ]add Turing DataFrames LinearAlgebra Distributions CategoricalArrays Random Optim StatsBase StatsPlots ProgressMeter DataFramesMeta Dates CSV Statistics FiniteDiff
using Turing
using DataFrames, DataFramesMeta
using LinearAlgebra
using Optim
using StatsBase  # for coeftable and stderror
using StatsPlots  # for histogram
using ProgressMeter
using Random
using Parameters
using FiniteDiff

module Model
    include("../Model/Model.jl")
end
include("HelperFunctions.jl")  # HF




#######################################################################
#                            Turing.jl Functions
#######################################################################

#######################################
#  Model functions
#######################################
# Karp model, AR(1) variances
@model function karp_model2(eᵢₜ, eₜ; ρ=missing)
    T = length(eₜ)
    N = length(eᵢₜ) ÷ T
    # Set variance priors
    σα² ~ InverseGamma(1, 1)
    σμ² ~ InverseGamma(1, 1)
    σθ² = σα² + σμ²/N

    # Set FE and time trend priors
    b₀ ~ MvNormal(zeros(N), 2)
    B₀ = mean(b₀)
    β₁ ~ truncated(Normal(0, 0.5); lower=0)
    β₂ ~ Normal(0, 0.1)

    # Set AR(1) coefficient prior
    if ismissing(ρ)
        ρ ~ truncated(Normal(0.87, 0.05); lower=0, upper=1)
    end

    # DGP models
    for t = 1:T
        # AR(1) error variances
        σₜ² = σθ²*sum(ρ^(2*s) for s=0:(t-1))
        σᵢₜ² = σμ²*(N-1)/N + σθ²*sum(ρ^(2*s) for s=0:(t-1))
        # Period Avg observation
        eₜ[t] ~ Normal(B₀ + β₁*t + β₂*t^2, sqrt(σₜ²))
        for i = 1:N
            # Period-unit observation
            eᵢₜ[i + (t-1)*N] ~ Normal(b₀[i] + β₁*t + β₂*t^2, sqrt(σᵢₜ²))
        end
    end

    return
end
# Karp model, AR(1) errors, informative priors
@model function karp_model3(eᵢₜ, eₜ; ρ=missing)
    T = length(eₜ)
    N = length(eᵢₜ) ÷ T
    # Set variance priors
    σα² ~ InverseGamma(1, 1)
    σμ² ~ InverseGamma(1, 1)

    # Set FE and time trend priors
    b₀ ~ MvNormal(zeros(N), 2)
    B₀ = mean(b₀)
    β₁ ~ truncated(Normal(0, 0.5); lower=0)
    β₂ ~ Normal(0, 0.1)

    # Set AR(1) coefficient prior
    if ismissing(ρ)
        ρ ~ truncated(Normal(0.87, 0.05); lower=0, upper=1)
    end

    # Initialize an empty vector to store the model AR(1) errors
    vₜ₋₁ = Vector{Real}(undef, T+1)
    # This has T+1 elements because we need to store vₜ₋₁[1] = 0
    vₜ₋₁[1] = 0

    # DGP models
    for t = 1:T
        # Period Avg observation
        eₜ[t] ~ Normal(B₀ + β₁*t + β₂*t^2 + ρ*vₜ₋₁[t], sqrt(σα² + σμ²/N))
        vₜ₋₁[t+1] = eₜ[t] - B₀ - β₁*t - β₂*t^2
        for i = 1:N
            # Period-unit observation
            eᵢₜ[i + (t-1)*N] ~ Normal(b₀[i] + β₁*t + β₂*t^2 + ρ*vₜ₋₁[t], sqrt(σα² + σμ²))
        end
    end

    return
end
# Karp model, AR(1) errors, flat priors
@model function karp_model4(eᵢₜ, eₜ; 
        ρ=missing, 
        σα²dist=InverseGamma(1, 1), 
        σμ²dist=InverseGamma(1, 1),
        ρdist=truncated(Normal(0.87, 0.05); lower=0, upper=1),
        b₀sd=2, β₁sd=0.5, β₂sd=0.1
    )
    T = length(eₜ)
    N = length(eᵢₜ) ÷ T
    # Set variance priors
    σα² ~ σα²dist
    σμ² ~ σμ²dist

    # Set FE and time trend priors
    b₀ ~ MvNormal(zeros(N), b₀sd)
    B₀ = mean(b₀)
    β₁ ~ truncated(Normal(0, β₁sd); lower=0)
    β₂ ~ Normal(0, β₂sd)

    # Set AR(1) coefficient prior
    if ismissing(ρ)
        ρ ~ ρdist
    end

    # Initialize an empty vector to store the model AR(1) errors
    vₜ₋₁ = Vector{Real}(undef, T+1)
    # This has T+1 elements because we need to store vₜ₋₁[1] = 0
    vₜ₋₁[1] = 0

    # DGP models
    for t = 1:T
        # Period Avg observation
        eₜ[t] ~ Normal(B₀ + β₁*t + β₂*t^2 + ρ*vₜ₋₁[t], sqrt(σα² + σμ²/N))
        vₜ₋₁[t+1] = eₜ[t] - B₀ - β₁*t - β₂*t^2
        for i = 1:N
            # Period-unit observation
            eᵢₜ[i + (t-1)*N] ~ Normal(b₀[i] + β₁*t + β₂*t^2 + ρ*vₜ₋₁[t], sqrt(σα² + σμ²))
        end
    end

    return
end

#######################################
#  Helper functions
#######################################
"""Generate outcome emission variables used in MLE from dataframe."""
function gen_MLE_data(data)
    eit = data.eᵢₜ
    et = combine(groupby(data, :t), :eᵢₜ => mean => :eₜ).eₜ
    return (; eit, et)
end

"""Return outcome emission variables used in MLE."""
function get_MLE_data(;N=4, T=60)
    # load real data, then transform eit to eit/1000
    data = @chain HF.read_data(N, T) @transform(:eᵢₜ = :eᵢₜ ./ 1000)
    return gen_MLE_data(data)
end
function get_MLE_data(θ; N=4, T=60)
    # simulate data based on approximate parameters recovered from data
    b₀ = [3.146, -0.454, 3.78, 3.479]
    β = [0.265, 0]
    data = Model.dgp(θ.ρ, θ.σₐ², θ.σᵤ², β, N, T; b₀ = b₀)
    return gen_MLE_data(data)
end

"""Return a dataframe of coefficients, SEs, and 95% CIs from the best run."""
function coef_df(res)
    # Create matrix of coefficients and standard errors
    #! Would want to use coeftable but there's a negative in the vcov matrix
    mat = [res.best_optim.values  sqrt.(abs.(diag(vcov(res.best_optim))))]
    # Create 95% confidence intervals
    LB = mat[:, 1] .- 1.96*mat[:, 2]
    UB = mat[:, 1] .+ 1.96*mat[:, 2]
    mat = [names(mat)[1] mat LB UB]
    # Add seed, LL, and convergence info to the matrix
    convergence = res.best_optim.optim_result.iteration_converged ? "Max Iterations" :
        res.best_optim.optim_result.f_converged ? "Function Convergence" :
        res.best_optim.optim_result.g_converged ? "Gradient Convergence" : "Unknown"
    mat = [mat
           :LL   res.best_run.LL   nothing nothing nothing
           :seed res.best_run.seed nothing nothing nothing
           :Iterations res.best_run.iters nothing nothing nothing
           :Termination convergence nothing nothing nothing ]
    df = DataFrame(mat, Symbol.(["Parameter", "Estimate", "Standard Erorr", "CI Lower 95", "CI Upper 95"]))
    return df
end

"""Plot the log likelihood surface over σ's from a chain and model"""
function plot_sampler(chain, model, param_values; label="", angle=(30, 65))
    # Evaluate log likelihood function at values of σs
    evaluate(σα², σμ²) = logjoint(model, 
        merge((;
            # σα²=invlink.(Ref(InverseGamma(2, 3)), σα²), 
            # σμ²=invlink.(Ref(InverseGamma(2, 3)), σμ²)),
            σα², 
            σμ²),
            param_values
        )
    )

    # Extract values from chain.
    val = get(chain, [:σα², :σμ², :lp])
    # σα² = link.(Ref(InverseGamma(2, 3)), val.σα²)
    # σμ² = link.(Ref(InverseGamma(2, 3)), val.σμ²)
    σα² = val.σα²
    σμ² = val.σμ²
    lps = val.lp

    # How many surface points to sample.
    granularity = 100

    # Range start/stop points.
    spread = 0.5
    σα²_start = minimum(σα²) - spread * std(σα²)
    σα²_stop = maximum(σα²) + spread * std(σα²)
    σμ²_start = minimum(σμ²) - spread * std(σμ²)
    σμ²_stop = maximum(σμ²) + spread * std(σμ²)
    σα²_rng = collect(range(σα²_start; stop=σα²_stop, length=granularity))
    σμ²_rng = collect(range(σμ²_start; stop=σμ²_stop, length=granularity))

    # Make surface plot.
    p = surface(
        σα²_rng,
        σμ²_rng,
        evaluate;
        camera=angle,
        #   ticks=nothing,
        colorbar=false,
        color=:inferno,
        title=label,
    )

    line_range = 1:length(σμ²)

    scatter3d(
        σα²[line_range],
        σμ²[line_range],
        lps[line_range];
        mc=:viridis,
        marker_z=lps[line_range],
        msw=0,
        legend=false,
        colorbar=false,
        alpha=0.5,
        xlabel="σα²",
        ylabel="σμ²",
        zlabel="Log probability",
        title=label,
        # camera=angle,
    )

    return p
end

"""Plot the log likelihood surface over σ's from model"""
function plot_LL(est_model, model; label="", angle=(30, 65), lower=missing, upper=missing, zlims=missing, granularity=100)

    param_values = (;est_model.best_run.ρ, 
                 b₀=[est_model.best_optim.values[Symbol("b₀[$i]")] for i in 1:4],
                 β₁=est_model.best_run.β1,
                 β₂=est_model.best_run.β2
    )
    # Evaluate log likelihood function at values of σs
    function evaluate(σα², σμ²)
        logjoint(model, merge((;
        # σα²=invlink.(Ref(InverseGamma(2, 3)), σα²), 
        # σμ²=invlink.(Ref(InverseGamma(2, 3)), σμ²)),
        σα², 
        σμ²),
        param_values
    ))
    end

    # Range start/stop points.
    if ismissing(lower) | ismissing(upper)
        df = coef_df(est_model)
    else
        df = nothing
    end
    if ismissing(lower)
        σα²_start = max(df[1,"CI Lower 95"],0)
        σμ²_start = max(df[2,"CI Lower 95"],0)
    elseif typeof(lower) <: Real
        σα²_start = lower
        σμ²_start = lower
    elseif typeof(lower) <: AbstractArray
        σα²_start = lower[1]
        σμ²_start = lower[2]
    end
    if ismissing(upper)
        σα²_stop = df[1,"CI Upper 95"]
        σμ²_stop = df[2,"CI Upper 95"]
    elseif typeof(upper) <: Real
        σα²_stop = upper
        σμ²_stop = upper
    elseif typeof(upper) <: AbstractArray
        σα²_stop = upper[1]
        σμ²_stop = upper[2]
    end
    if ismissing(zlims)
        zlims = (nothing, nothing)
    end
    x1, x2 = log.([σα²_start, σα²_stop])
    σα²_rng = exp.(range(x1; stop=x2, length=granularity))
    σμ²_rng = collect(range(σμ²_start; stop=σμ²_stop, length=granularity))


    # Make surface plot.
    p = surface(
        σα²_rng,
        σμ²_rng,
        evaluate;
        camera=angle,
        #   ticks=nothing,
        colorbar=false,
        color=:inferno,
        title=label,
        xlabel="σα²",
        ylabel="σμ²",
        zlabel="Log probability",
        zlims=zlims,
    )

    return p
end

"""Return starting value used in optim.
    Requires optim be run with store_trace=true and extended_trace=true
"""
function get_iteration_x(optim_result, iter)
    if iter == "start"
        i = 1
    elseif iter == "end"
        i = length(Optim.x_trace(optim_result))
    else
        i = iter
    end
    # get starting x from trace
    x0 = Optim.x_trace(optim_result)[i]
    # apply appropriate transforms for each parameter
    x0 = [ℯ.^x0[1:2]; x0[3:6]; ℯ^x0[7]; x0[8]; cdf(Normal(), x0[end])]
    return x0
end
"""Return dictionary of starting parameter values used in optim for each seed.
    Requires optim be run with store_trace=true and extended_trace=true
    iter = "start" is the starting search parameter value
    iter = "end" is the final search parameter value
"""
function get_iteration_x_dict(opt_dict; iter="start")
    x0_dict = Dict()
    for seed in keys(opt_dict)
        x0_dict[seed] = get_iteration_x(opt_dict[seed].optim_result, iter)
    end
    return x0_dict
end


#######################################
# Estimation functions
#######################################
function model2_few_params()

    # Get the simulated data
    # simulate data based on approximate parameters recovered from data
    b₀ = [3.146, -0.454, 3.78, 3.479]
    β = [0.265, 0]
    data = Model.dgp(θ.ρ, θ.σₐ², θ.σᵤ², β, N, T; b₀ = b₀)
    eit = data.eᵢₜ
    et = combine(groupby(data, :t), :eᵢₜ => mean => :eₜ).eₜ

    # Estimate the MLE
    opt4 = optimize(karp_model2(eit, et), MLE())
    opt4.lp
    opt4.values
    opt4.optim_result
    # Then, get the standard errors
    coefdf = coeftable(opt4)
    true_params

    # Try again and fix ρ
    # Initialize dataframe to store the results
    df = DataFrame(UID=Int64[], LL=Float64[], iters=Float64[], 
                    σα²=Float64[], σμ²=Float64[], ρ=Float64[],
                    b01=Float64[], b02=Float64[], b03=Float64[], b04=Float64[],
                    β1=Float64[], β2=Float64[],
                iconverge=Bool[], gconverge=Bool[], fconverge=Bool[],
                gtol=Float64[], ftol=Float64[],
                N=Int64[], T=Int64[], ρfixed=Bool[]
    ); optim_dict = Dict(); UID = 1
    ftol = 1e-40

    # Estimate the MLE
    @showprogress for _ in 1:10
        opt5 = optimize(
            karp_model2(eit, et; ρ=θ.ρ), MLE(), 
            ConjugateGradient(),
            Optim.Options(iterations=50_000, g_tol = 1e-12, f_tol=
            store_trace = true, show_trace=false)
        );
        @show push!(df, (UID, opt5.lp,  # Log likelihood value
            opt5.optim_result.iterations,  # number of iterations
            opt5.values[:σα²], opt5.values[:σμ²], ρ,
            opt5.values[Symbol("b₀[1]")], opt5.values[Symbol("b₀[2]")], opt5.values[Symbol("b₀[3]")], opt5.values[Symbol("b₀[4]")],
            opt5.values[:β₁], opt5.values[:β₂],
            opt5.optim_result.iteration_converged,  # did it hit max iterations?
            opt5.optim_result.g_converged, # did it converge from the gradient?
            opt5.optim_result.f_converged,  # did it converge from the LL value?
            opt5.optim_result.g_abstol,  # gradient tolerance setting
            length(eit) ÷ length(et), length(et), true,  # N, T, ρ is fixed
        ))
        optim_dict[UID] = opt5
        UID += 1
    end

    coeftable(optim_dict[6])
    true_params

end

"""Return model initialized with data.
    
    `initialize_model(karp_model4, false, missing, "real")`
                      turing model, ρfixed, θ,     datatype
    If datetype is "real", initialize with real data. Otherwise, initialize with simulated data.
    If ρfixed is true, use θ.ρ as the fixed ρ value in initializing the model. 
        In this case, θ requires at least the field θ.ρ.
    If ρfixed is false, use `missing`, which estimates ρ.
        In this case, use θ = missing since the true parameters are to be estimated.
"""
function initialize_model(model, ρfixed, θ, datatype; kwargs...)
    # Get real or simulated data
    Y = datatype=="real" ? get_MLE_data() : get_MLE_data(θ)
    # Initialize the model
    ρ = ρfixed ? θ.ρ : missing
    return model(Y.eit, Y.et; ρ=ρ, kwargs...)
end
# Run immediately to precompile

"""Run MLE on model, storing results in df and dict."""
function multistart_MLE(m, Nseeds; maxiter=maxiter)
    # Initialize list, dictionary, and unique ID to store the results from each estimation
    dfs = Array{DataFrame}(undef, Nseeds)
    dict = Dict{Int64, Turing.TuringOptimExt.ModeResult}()
    ρfixed = !ismissing(m.defaults.ρ)

    # run optimize on the model `seeds` times, adding summary results to df
    # storing the full results in a dictionary indexed by UID
    p = Progress(Nseeds)
    Threads.@threads for seed in 1:Nseeds
    # for seed in 1:Nseeds
        Random.seed!(seed)
        # Estimate the model
        #! Handle convergence failure warning
        #! See Optim.converged(M)
        opt = optimize(m, MLE(), ConjugateGradient(),
                       Optim.Options(iterations=maxiter,
                                     store_trace=true,
                                     extended_trace=true
                       )
        )
        # Store the results
        df = DataFrame(
            seed=seed,
            LL=opt.lp,  # Log likelihood value
            iters=opt.optim_result.iterations,  # number of iterations
            σα²=opt.values[:σα²],  # Estimated parameters
            σμ²=opt.values[:σμ²],
            ρ= ρfixed ? m.defaults.ρ : opt.values[:ρ],
            b01=opt.values[Symbol("b₀[1]")],
            b02=opt.values[Symbol("b₀[2]")],
            b03=opt.values[Symbol("b₀[3]")],
            b04=opt.values[Symbol("b₀[4]")],
            β1=opt.values[:β₁],
            β2=opt.values[:β₂],
            iconverge=opt.optim_result.iteration_converged,  # did it hit max iterations?
            gconverge=opt.optim_result.g_converged, # did it converge from the gradient?
            fconverge=opt.optim_result.f_converged,  # did it converge from the LL value?
            gtol=opt.optim_result.g_abstol,  # gradient tolerance setting
            N=length(m.args.eᵢₜ) ÷ length(m.args.eₜ),
            T=length(m.args.eₜ),
            ρfixed=!ismissing(m.defaults.ρ),  # is ρ fixed in the model
        )
        dfs[seed] = df
        dict[seed] = opt
        next!(p)
    end
    df = vcat(dfs...)
    return (; df, dict)
end
# Run immediately to precompile
_m = initialize_model(karp_model4, false, missing, "real");
multistart_MLE(_m, 1; maxiter=2);

"""Run MLE on model, returning a dataframe of results.

    `estimate_MLE(karp_model3, ρfixed=false, θ=missing, maxiter=10_000, seeds=100)` will 
        estimate karp_model3 on real data, 100 times, including estimating ρ.
    If datetype is "real", estimate with real data. Otherwise, estimate with simulated data.
    If datetype is "simulated", θ must be a named tuple with attributes: θ.ρ, θ.σₐ², θ.σᵤ²

    - ρ fixed, simulated data, must give θ
    - ρ fixed, real data, must give θ
    - ρ free, simulated data, must give θ
    - ρ free, real data, can leave θ missing
"""
function estimate_MLE(model; ρfixed=false, θ=missing, datatype="real", maxiter=10_000, Nseeds=100, kwargs...)
    # Initialize the model with data
    println("\nInitializing model with $datatype data and ρfixed=$ρfixed")
    model_obj = initialize_model(model, ρfixed, θ, datatype; kwargs...)

    # Run MLE with different random seeds to get multiple starting points
    println("\nRuning Multistart Search for MLE")
    ms_result = multistart_MLE(model_obj, Nseeds; maxiter=maxiter)

    # Find best run (highest LL), and return that result
    println("\nFinding best run.")
    best_run = ms_result.df[findmax(ms_result.df.LL)[2], :]
    best_optim = ms_result.dict[best_run.seed]
    return (; best_run, best_optim, ms_result)
end

function convert_params_to_namedtuple(nv::NamedVector)
    # nv is a named vector of parameters
    nt = (
        σα² = nv[:σα²],
        σμ² = nv[:σμ²],
        b₀ = [nv[Symbol("b₀[$i]")] for i in 1:4],
        β₁ = nv[:β₁],
        β₂ = nv[:β₂],
        ρ = nv[:ρ],
    )
end
function convert_params_to_namedtuple(v::Vector) 
    # v is a vector of parameters in the following order
    nt = (
        σα² = v[1],
        σμ² = v[2],
        b₀ = [v[i+2] for i in 1:4],
        β₁ = v[7],
        β₂ = v[8],
        ρ = v[9],
    )
end


###############################################
# Test the optimization function and output
m1 = initialize_model(karp_model3, false, missing, "real")
mle_estimate = optimize(m1, MLE(), ConjugateGradient(),
               Optim.Options(iterations=10_000, store_trace=true, extended_trace=true)
)
coeftable(mle_estimate) #! does not work: 
mle_estimate.optim_result.iterations

# Standard errors are calculated from the Fisher information matrix (inverse Hessian of the log likelihood or log joint)
# Calculate the Hessian
# Need the loglikelihood function as a function of one vector of parameters
θmle = convert_params_to_namedtuple(mle_estimate.values)
logjoint(m1, θmle)
LL(θ) = logjoint(m1, convert_params_to_namedtuple(θ))
# Calculate the finite difference numerical hessian at the MLE location
FiniteDiff.finite_difference_hessian(LL, mle_estimate.values.array)
#! domain error here -- probably because of the σa parameter being close to 0.
#! try again with σa being larger
θmle2 = mle_estimate.values.array+[0.00013, 0, 0, 0, 0, 0, 0, 0, 0];
H = FiniteDiff.finite_difference_hessian(LL, θmle2);
VCOV = inv(-H);
se = sqrt.(diag(VCOV))

###############################################
# Test posterior chains
# Sample with the MAP estimate as the starting point.
chain = sample(m1, NUTS(), 1_000; discard_initial = 1000, init_params = mle_estimate.values.array)
plot(chain)
#! check the convergence on this, check discard amount, plot dists of parameters, get 95% CI

# Multiple chains
chain = sample(m1, NUTS(), MCMCThreads(), 1_000, 4; init_params = mle_estimate.values.array)
chain = sample(m1, NUTS(), MCMCThreads(), 1_000, 4; discard_initial = 1000, init_params = mle_estimate.values.array)
#! Above parallel sampling does not work, strange error.
# chn = sample(gdemo(1.5, 2), NUTS(), MCMCThreads(), 1000, 4; discard_initial = 1000)
# Replace num_chains below with however many chains you wish to sample.
chains = mapreduce(c -> sample(m1, NUTS(), 1_000; discard_initial = 1000, init_params = mle_estimate.values.array), chainscat, 1:4)
chains.value[Axis{:var}][10]  # lp
minimum(chains.value[:,10,:])
#! Don't seem to get a good lp as optim. Even the min doesn't get close
#! I wonder if this is just the LL or does it include the prior?
#! how to do this in parallel?
# https://stackoverflow.com/questions/51228889/how-do-i-do-parallel-mapreduce-in-julia
# ThreadsX.mapreduce https://github.com/tkf/ThreadsX.jl
# pmapreduce https://docs.juliahub.com/ParallelUtilities/SO4iL/0.8.6/pmapreduce/

plot(chains)


###############################################  ESIMATE σα², σμ², ρ
# Estimate the MLE on real data, estimate ρ
result_freeρ = estimate_MLE(karp_model3; ρfixed=false, datatype="real", maxiter=50, Nseeds=20)

# Retrieve starting parameter values from multistarted results
x0_dict = get_iteration_x_dict(result_freeρ.ms_result.dict)
density([x0_dict[i][1] for i in keys(x0_dict)], 
    label="σα²", bins=length(x0_dict), bandwidth=0.1)
sort([x0_dict[i][1] for i in keys(x0_dict)])

# min parallel for Nseeds=5,  7:42 min parallel for Nseeds=20
@chain result_freeρ.ms_result.df @orderby(:LL)
result_freeρ.best_run
coef_df(result_freeρ)
plot_LL(result_freeρ, m1; angle=(30, 40), lower=0, upper=1)
plot_LL(result_freeρ, m1; angle=(30, 40), lower=[1e-10,0.398], upper=[0.1, 0.4], zlims=[-320, -230])

# Access the Turing chain and explore the posterior distribution for σα² and σμ²
c = Turing.sample(m1, NUTS(), MCMCThreads(), 1000, 10; discard_initial=1000)
# Condition on the other parameters fitted values
param_values = (;result_freeρ.best_run.ρ, 
                 b₀=[result_freeρ.best_optim.values[Symbol("b₀[$i]")] for i in 1:4],
                 β₁=result_freeρ.best_run.β1,
                 β₂=result_freeρ.best_run.β2
)
# Plot the posterior distribution
plot_sampler(c, m1, param_values; angle=(30, 45))
# Rotate the posterior distribution to get a better look
anim = @animate for vert=10:20:90, rad=10:10:360
    @show rad, vert
    plot_sampler(c, m1, param_values; angle=(rad, vert))
end
gif(anim, "LL_turing_NUTSsampler_points.gif", fps = 10)



# Experiment with different priors on σα² and σμ²  (est runtime = 5 min)
#! increase maxiter: at least 4 don't converge
#! increase Nseeds: check more of the parameter space
result_freeρ_uniform1 = estimate_MLE(karp_model4; ρfixed=false, datatype="real", maxiter=50_000, Nseeds=20,
    σα²dist=Uniform(0, 1e10)
)  # 7 min
result_freeρ_uniform2 = estimate_MLE(karp_model4; ρfixed=false, datatype="real", maxiter=50_000, Nseeds=20,
    σα²dist=Uniform(0, 1e10),
    σμ²dist=Uniform(0, 1e10),
)  # 5 min
result_freeρ_uniform3 = estimate_MLE(karp_model4; ρfixed=false, datatype="real", maxiter=50_000, Nseeds=20,
    σα²dist=Uniform(0, 1e10),
    σμ²dist=Uniform(0, 1e10),
    ρdist=Uniform(-1, 1)
)  # 4 min
result_freeρ_uniform4 = estimate_MLE(karp_model4; ρfixed=false, datatype="real", maxiter=50_000, Nseeds=20,
    σα²dist=Uniform(0, 1e10),
    σμ²dist=Uniform(0, 1e10),
    ρdist=Uniform(-1, 1),
    b₀sd=20, β₁sd=5, β₂sd=1
)
result_freeρ_uniform1.best_run
result_freeρ_uniform2.best_run
result_freeρ_uniform3.best_run
result_freeρ_uniform4.best_run
coef_df(result_freeρ_uniform1)
coeftable(result_freeρ_uniform4.best_optim)
@chain result_freeρ_uniform1.ms_result.df @orderby(:LL)

# Examine the starting parameter values
x0_dict2 = get_iteration_x_dict(result_freeρ_uniform1.ms_result.dict)
xend_dict2 = get_iteration_x_dict(result_freeρ_uniform1.ms_result.dict; iter="end")
density([log10(x0_dict2[i][1]) for i in keys(x0_dict2)], 
    label="σα²", bandwidth=0.1)
sort([x0_dict2[i][1] for i in keys(x0_dict2)])
sort([x0_dict2[i][2] for i in keys(x0_dict2)])
sort([x0_dict2[i][end] for i in keys(x0_dict2)])

# Plot the starting parameter values for σα² and σμ²
_x, _y = vcat.([x0_dict2[i][1:2] for i in keys(x0_dict2)]...)
scatter(_x, _y)


"""
Parameter estimates for real data / 1000
    A ╲ hcat │       est         SE          95% LB        95% UB
    ─────────┼───────────────────────────────────────────────────
    σα²      │ 3.59336e-19    0.0246159   -0.0482472    0.0482472
    σμ²      │    0.399063    0.0301746     0.339921     0.458205
    b₀[1]    │     1.23016     0.247622     0.744824       1.7155
    b₀[2]    │  -0.0891669     0.247622    -0.574506     0.396172
    b₀[3]    │     1.46239     0.247622     0.977054      1.94773
    b₀[4]    │     1.35187     0.247622     0.866527      1.83721
    β₁       │   0.0903224     0.035658    0.0204327     0.160212
    β₂       │  8.49059e-5  0.000633762  -0.00115727   0.00132708
    ρ        │    0.929347     0.124367     0.685588      1.17311
- robust to all checks on the priors

Parameter estimates for raw data
    9-element Named Vector{Float64}
    A     │       est   │  SE (seems wrong)
    ──────┼──────────  ─┼───────────
    σα²   │       0.0   │    1.23397
    σμ²   │ 3.99063e5   │    1.52001
    b₀[1] │   1230.16   │    1.10942
    b₀[2] │  -89.1669   │    1.10942
    b₀[3] │   1462.39   │    1.10942
    b₀[4] │   1351.87   │    1.10942
    β₁    │   90.3224   │  0.0884016
    β₂    │ 0.0849059   │ 0.00147306
    ρ     │  0.929347   │  0.0489734

"""




##########################################################################################
#                        Test on simulated data over a range of parameters
##########################################################################################
# Define range of parameters
#! should I just generate them from turing model?

# Define number of steps in each parameter space

# Define linear or log steps for each parameter

# Generate full list of all parameter combinations

# Run estimation for each parameter combination
# Run each parameter combination Nsim times with different seeds for random data generation





############################## try with many periods
# Generate simulated data with many time periods
T3 = 200
data3 = Model.dgp(θ.ρ, θ.σₐ², θ.σᵤ², β, N, T3; b₀ = b₀)
histogram(data3.eᵢₜ, label="eᵢₜ", title="Simulated data with trends", legend=:topleft)

@showprogress for _ in 1:10
    @time opt6 = optimize(
        karp_model2(data3.eᵢₜ, combine(groupby(data3, :t), :eᵢₜ => mean => :eₜ).eₜ; ρ=θ.ρ), 
        MLE(), ConjugateGradient(),
        Optim.Options(iterations=50_000, g_tol = 1e-12, store_trace = false, show_trace=false)
    )
    @show push!(df3, (opt6.lp,  # Log likelihood value
                 opt6.optim_result.iterations,  # number of iterations
                 opt6.values[1], opt6.values[2], θ.ρ,  # estimated parameters
                 opt6.optim_result.iteration_converged,  # did it hit max iterations?
                 opt6.optim_result.g_converged, # did it converge from the gradient?
                 opt6.optim_result.f_converged,  # did it converge from the LL value?
                 opt6.optim_result.g_abstol,  # gradient tolerance set
                 4, T3,  # N and T
    ))
    push!(optim_list3, opt6)
end
coeftable(opt6)


#######################################
#  Turing attempt for Karp model, adding time averaging
#######################################
#! Add time averaging to this model
@model function karp_model5(eᵢₜ, eₜ; ρ=nothing)
    T = length(eₜ)
    N = length(eᵢₜ) ÷ T
    # Set variance priors
    σα² ~ InverseGamma(1, 1)
    σμ² ~ InverseGamma(1, 1)

    # Set FE and time trend priors
    b₀ ~ MvNormal(zeros(N), 2)
    B₀ = mean(b₀)
    β₁ ~ truncated(Normal(0, 0.5); lower=0)
    β₂ ~ Normal(0, 0.1)

    # Set AR(1) coefficient prior
    if isnothing(ρ)
        ρ ~ truncated(Normal(0.87, 0.05); lower=0, upper=1)
    end

    # DGP models
    for t = 1:T
        # AR(1) error variances
        σₜ² = (σα² + σμ²/N)*sum(ρ^(2*(s-1)) for s=1:t)
        σᵢₜ² = (σα² + σμ²)*sum(ρ^(2*(s-1)) for s=1:t)
        println("Period t=$t: 2*(s-1)")
        # Period Avg observation
        eₜ[t] ~ Normal(B₀ + β₁*t + β₂*t^2, sqrt(σₜ²))
        for i = 1:N
            # Period-unit observation
            eᵢₜ[i + (t-1)*N] ~ Normal(b₀[i] + β₁*t + β₂*t^2, sqrt(σᵢₜ²))
        end
    end

    return
end
#! rename variables below
# Get the simulated data
eit = data2.eᵢₜ
et = combine(groupby(data2, :t), :eᵢₜ => mean => :eₜ).eₜ
# Estimate the MLE
opt6 = optimize(karp_model2(eit, et), MLE())
opt6.lp
opt6.values
opt6.optim_result
# Then, get the standard errors
coefdf = coeftable(opt6)
stderror(opt6)

##########################################################################################
#                                   Examples
##########################################################################################

#######################################
#     Example MLE turing model from
# https://discourse.julialang.org/t/get-mle-parameters-e-g-p-value-confidence-intervals-from-a-turing-model-using-optim-jl/101433?u=acwatt
#######################################
using Turing, DataFrames, LinearAlgebra, Optim, StatsBase
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [0, 0.6, 1, 1.4, 2, 2.8, 3, 3.3, 4, 4.6]

@model function lm(y, x)
    # Set variance prior.
    σ² ~ truncated(Normal(0, 100); lower=0)
    intercept ~ Normal(0, sqrt(3))
    coefficient ~ TDist(3)

    # Calculate all the mu terms.
    mu = intercept .+ x * coefficient
    y ~ MvNormal(mu, σ² * LinearAlgebra.I)
end

# Estimate the MLE
opt = optimize(lm(y,x), MLE())
# Then, get the standard errors
infomat = stderror(opt)
coefdf = coeftable(opt)

#######################################
#     Example AR(2) turing model from
# https://stackoverflow.com/questions/67730294/typeerror-when-sampling-ar2-process-using-turing-jl
#######################################
using Statistics
using Turing, Distributions


@model function AR2(y, b1_mean, b2_mean; σ=.3, σ_b=.01)
    N = length(y)
           
    b1 ~ Normal(b1_mean, σ_b)
    b2 ~ Normal(b2_mean, σ_b)
           
    y[1] ~ Normal(0, σ)
    y[2] ~ Normal(0, σ)
           
    for k ∈ 3:N
        y[k] ~ Normal(b1 * y[k - 1] + b2 * y[k - 2], σ) # this is LINE 19
    end
           
    y
end


# These are the parameters
b1, b2 = .3, .6

println("Constructing random process with KNOWN parameters: b1=$b1, b2=$b2")
AR2_process = AR2(fill(missing, 500), b1, b2)()
AR2_process_orig = copy(AR2_process)

@show typeof(AR2_process)

println("Estimating parameters...")
model_ar = AR2(AR2_process, b1, b2)
chain_ar = sample(model_ar, HMC(0.001, 10), 200)

@show mean(chain_ar[:b1]), mean(chain_ar[:b2])
@show all(AR2_process .≈ AR2_process_orig)

println("Now try to estimate Float64")
model_ar = AR2(Float64.(AR2_process), b1, b2)
chain_ar = sample(model_ar, HMC(0.001, 10), 200)

@show mean(chain_ar[:b1]), mean(chain_ar[:b2])

# Estimate the MLE
opt2 = optimize(AR2(AR2_process, b1, b2), MLE())
# Then, get the standard errors
infomat = stderror(opt2)
coefdf = coeftable(opt2)

