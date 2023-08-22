
println(@__DIR__)
using Pkg; Pkg.activate(joinpath(@__DIR__, "turing_test_env"))
# ]add Turing DataFrames LinearAlgebra Distributions CategoricalArrays Random Optim StatsBase StatsPlots ProgressMeter DataFramesMeta Dates CSV Statistics FiniteDiff JLD
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
using NamedArrays
using Printf
using JLD  # for saving objects as .jld files

module Model
    include("../Model/Model.jl")
end
include("HelperFunctions.jl")  # HF
include("../Model/TuringModels.jl")  # TuringModels




#######################################################################
#                            Turing.jl Functions
#######################################################################


#######################################
#  Helper functions
#######################################
    """Generate outcome emission variables used in MLE from dataframe."""
function gen_MLE_data(data)
    eᵢₜ = data.eᵢₜ
    eₜ = combine(groupby(data, :t), :eᵢₜ => mean => :eₜ).eₜ
    return (; eᵢₜ, eₜ)
end

"""If `field` is not in NamedTuple `nt`, set it to `default_value`."""
function set_param_field(nt, field, default_value)
    if !isin(field, nt) || ismissing(getfield(nt, field))
        nt = merge(nt..., (; field => default_value))
    end
    return nt
end

"""Return outcome emission variables used in MLE."""
function get_MLE_data(; N=4, T=60)
    println("get_MLE_data observed data")
    # load real data, then transform eit to eit/1000
    data = @chain HF.read_data(N, T) @transform(:eᵢₜ = :eᵢₜ ./ 1000)
    return gen_MLE_data(data)
end
function get_MLE_data(θ; N=4, T=60)
    println("get_MLE_data with θ")
    # simulate data based on approximate parameters recovered from data
    b₀ = isin(:b₀, θ) ? θ.b₀ : [3.146, -0.454, 3.78, 3.479]
    β1 = isin(:β₁, θ) ? θ.β₁ : 0.265
    β2 = isin(:β₂, θ) ? θ.β₂ : 0
    β = [β1; β2]
    v₀ = isin(:v0, θ) ? θ.v0 : 0.0
    
    data = Model.dgp(θ.ρ, θ.σα², θ.σμ², β, N, T; b₀, v₀)
    return gen_MLE_data(data)
end
function get_MLE_data(model, θ; N=4, T=60, seed=1234)
    println("get_MLE_data with model")
    # if any parameter is missing, use approximate parameters recovered from data
    θ = set_param_field(θ, :b₀, [3.146, -0.454, 3.78, 3.479])
    θ = set_param_field(θ, :β₁, 0.265)
    θ = set_param_field(θ, :β₂, 0)
    θ = set_param_field(θ, :v0, 0.0)
    # Initialize DGP model
    m = model(missing, θ; N, T)
    # Sample the DGP
    Random.seed!(seed)
    data = m().Y
    return data
end

"""Return a dataframe of coefficients, SEs, and 95% CIs from the best run."""
function coef_df(res)
    # Create matrix of coefficients and standard errors
    #! Would want to use coeftable but there's a negative in the vcov matrix
    mat = [res.best_optim.values  sqrt.(abs.(diag(vcov(res.best_optim))))]
    # Create 95% confidence intervals
    LB = mat[:, 1] .- 1.96*mat[:, 2]
    UB = mat[:, 1] .+ 1.96*mat[:, 2]
    mat = [names(mat)[1] mat LB UB ]
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
    df[!, "Is Parameter"] = Integer.([ones(length(res.best_optim.values)); zeros(4)])
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

function convert_params_to_namedtuple(nv::NamedVector)
    # nv is a named vector of parameters
    nt = (
        σα² = nv[Symbol("θ.σα²")],
        σμ² = nv[Symbol("θ.σμ²")],
        b₀ = [nv[Symbol("θ.b₀[$i]")] for i in 1:4],
        β₁ = nv[Symbol("θ.β₁")],
        β₂ = nv[Symbol("θ.β₂")],
        ρ = nv[Symbol("θ.ρ")],
    )
    if any(occursin.("v0", string.(names(nv)[1])))
        nt = (nt..., v0 = nv[Symbol("θ.v0")])
    end
    return nt
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
    if length(v) == 10
        nt = (nt..., v0 = v[10])
    end
    return nt
end
function convert_params_to_df(nt::NamedTuple)
    # return a named tuple
    # for each value in the named tuple, if it is a vector, break it into floats
    nt2 = (;)
    for k in keys(nt)
        v = nt[k]
        if typeof(v) <: AbstractArray
            for i in eachindex(v)
                nt2 = (nt2..., Symbol("θ.$k[$i]") => v[i])
            end
        else
            nt2 = (nt2..., Symbol("θ.$k") => v)
        end
    end
    df = DataFrame(nt2)
    rename!(df, :first => :Parameter, :second => :Truth)
    return df
end

"""Return true if s occurs in the keys of nt"""
function isin(s::Symbol, nt::NamedTuple)
    return any(occursin.(string(s), string.(keys(nt))))
end





#######################################
# Estimation functions
#######################################

"""
    `initialize_model(model; θ=missing, seed=1234, kwargs...)`
    `initialize_model(model, Prior::Turing.Prior; kwargs...)`

Return model initialized with data, or for prior sampling.

`initialize_model(model; θ=missing, kwargs...)`
- `model` = Turing model function
- `θ` = NamedTuple, parameters to generate simulated data with
  - If `θ = missing`, observed data is used, parameters are to be estimated.


# Examples
## Estimate MLE parameters of karp_model5 using observed data
```julia
m = initialize_model(TuringModels.karp_model5)
opt = optimize(m, MLE(), ConjugateGradient())
opt.values
```

## Set "true" parameters to generate data
```julia
θ = (; b₀=[0.1, 0.1, 0.1, 0.1], β₁=0.1, β₂=0.1, σα²=0.1, σμ²=0.1, ρ=0.1, v0=0)
# Generate simulated data and estimate MLE parameters of karp_model5
m = initialize_model(TuringModels.karp_model5; θ)
opt = optimize(m, MLE(), ConjugateGradient())
opt.values
```

## Sample data and parameters from prior
```julia
# Generate simulated data and estimate MLE parameters of karp_model5
m = initialize_model(TuringModels.karp_model5, Prior())
data = m()
```
"""
function initialize_model(model; θ=missing, seed=1234, kwargs...)
    # If θ is missing, Y = observed data, else Y = simulated data from θ
    Y = ismissing(θ) ? get_MLE_data() : get_MLE_data(model, θ; seed)
    return model(Y, missing; kwargs...)
end
function initialize_model(model, Prior::Turing.Prior; kwargs...)
    Y, θ = missing, missing
    return model(Y, θ; kwargs...)
end

"""Run MLE on model, storing results in df and dict."""
function multistart_MLE(m, Nseeds; maxiter=100_000)
    # Initialize list, dictionary, and unique ID to store the results from each estimation
    dfs = Array{DataFrame}(undef, Nseeds)
    dict = Dict{Int64, Turing.TuringOptimExt.ModeResult}()
    ρfixed = ismissing(m.args.θ) ? false : !ismissing(m.args.θ.ρ)

    # run optimize on the model `seeds` times, adding summary results to df
    # storing the full results in a dictionary indexed by UID
    # p = Progress(Nseeds)
    # Threads.@threads for seed in 1:Nseeds
    for seed in 1:Nseeds
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
        params = convert_params_to_namedtuple(opt.values)
        # Store the results
        df = DataFrame(
            seed=seed,
            LL=opt.lp,  # Log likelihood value
            iters=opt.optim_result.iterations,  # number of iterations
            σα²=params[:σα²],  # Estimated parameters
            σμ²=params[:σμ²],
            ρ=ρfixed ? m.defaults.θ.ρ : params[:ρ],
            b01=params[:b₀][1],
            b02=params[:b₀][2],
            b03=params[:b₀][3],
            b04=params[:b₀][4],
            β1=params[:β₁],
            β2=params[:β₂],
            v0=params[:v0],
            iconverge=opt.optim_result.iteration_converged,  # did it hit max iterations?
            gconverge=opt.optim_result.g_converged, # did it converge from the gradient?
            fconverge=opt.optim_result.f_converged,  # did it converge from the LL value?
            gtol=opt.optim_result.g_abstol,  # gradient tolerance setting
            N=length(m.args.Y.eᵢₜ) ÷ length(m.args.Y.eₜ),
            T=length(m.args.Y.eₜ),
            ρfixed=ρfixed,  # is ρ fixed in the model
        )
        dfs[seed] = df
        dict[seed] = opt
        # next!(p)
    end
    df = vcat(dfs...)
    return (; df, dict)
end

"""
    `estimate_MLE(model; θ=missing, maxiter=100_000, Nseeds=100, kwargs...)`
    `estimate_MLE(model, Nsim=100; θ=missing, maxiter=100_000, Nseeds=100, kwargs...)`

    Run MLE on model, returning a dataframe of results.

    `estimate_MLE(karp_model3, ρfixed=false, θ=missing, maxiter=10_000, Nseeds=100)` will 
        estimate karp_model3 on real data, 100 times, including estimating ρ.
    If datetype is "real", estimate with real data. Otherwise, estimate with simulated data.
    If datetype is "simulated", θ must be a named tuple with attributes: θ.ρ, θ.σₐ², θ.σᵤ²

    - ρ fixed, simulated data, must give θ
    - ρ fixed, real data, must give θ
    - ρ free, simulated data, must give θ
    - ρ free, real data, can leave θ missing
    - `Nseeds` : number of optim multistart runs to do from different Random seeds

    If `Nsim` is given, `Nsim` datasets will be generated from θ, and aggregate results will be returned. 
"""
function estimate_MLE(model; θ=missing, maxiter=100_000, Nseeds=100, kwargs...)
    # Initialize the model with data
    datatype = ismissing(θ) ? "observed" : "simulated"
    println("\nInitializing model with $datatype data")
    model_obj = initialize_model(model; θ, kwargs...)

    # Run MLE with different random seeds to get multiple starting points
    println("\nRuning Multistart Search for MLE")
    ms_result = multistart_MLE(model_obj, Nseeds; maxiter)

    # Find best run (highest LL), and return that result
    println("\nFinding best run.")
    best_run = ms_result.df[findmax(ms_result.df.LL)[2], :]
    best_optim = ms_result.dict[best_run.seed]
    return (; best_run, best_optim, ms_result)
end
function estimate_MLE(model, Nsim; θ=missing, maxiter=100_000, Nseeds=100, kwargs...)
    # Generate Nsim datasets from θ
    dfs = []
    p = Progress(Nsim)
    Threads.@threads for seed in 1:Nsim
        # Initialize the model with simulated data, generated from θ
        model_obj = initialize_model(model; θ, seed, kwargs...)
        # Run MLE with different random seeds to get multiple starting points
        println("\nRuning Multistart Search for MLE")
        ms_result = multistart_MLE(model_obj, Nseeds; maxiter)

        # Find best run (highest LL), and return that result
        println("\nFinding best run.")
        best_run = ms_result.df[findmax(ms_result.df.LL)[2], :]
        best_optim = ms_result.dict[best_run.seed]
        res = (; best_run, best_optim, ms_result)
        # Estimates and standard errors
        df = coef_df(res)
        # Add a column for the seed
        df.data_seed .= seed
        df.optim_Nseeds .= Nseeds
        push!(dfs, df)
        next!(p)
    end
    # True parameters
    df_truth = convert_params_to_df(θ)
    # Aggregate results
    df = vcat(dfs...)
    df = @chain df begin
        @subset($"Is Parameter" .== 1)
        leftjoin(df_truth, on=:Parameter)
        @transform(:bias = :Estimate - :Truth)
        @select(:data_seed, :Parameter, :bias)
        rightjoin(df, on=[:Parameter, :data_seed])
    end
    # Summary statistics of simulations
    df2 = @chain df begin
        groupby(:Parameter)
        @combine(
            :mean_estimate = mymean(:Estimate),
            :mean_se = mymean($"Standard Erorr"),
            :mean_bias = mean(:bias), 
            :mean_abs_bias = mean(abs.(:bias)),
            :var_bias = var(:bias),
            :var_abs_bias = var(abs.(:bias)),
        )
        leftjoin(df_truth, on=:Parameter)
        @transform(:Ndata_seeds = Nsim, :Noptim_seeds=Nseeds)
    end

    return (; df_results=df, df_summary=df2)
end

function mymean(x)
    if isa(x[1], Number)
        return mean(x)
    elseif isa(x[1], String)
        m = mean(x .!== "Max Iterations")*100
        return "$(@sprintf("%.1f", m))% converged"
    end
end

function test_MLE_simulation_loop()
    param_sampler = TuringModels.karp_model5_parameters(missing;
        σα²dist=Exponential(1),
        σμ²dist=Exponential(1)
    )
    _θ = param_sampler()
    _r =  estimate_MLE(TuringModels.karp_model5, 2; θ=_θ, maxiter=100_000, Nseeds=50)
    _r.df_summary
end

function test_models()
    println("Running test of simulated data MLE")
    # Generate parameters from prior
    param_sampler = TuringModels.karp_model5_parameters(missing)
    θ = param_sampler()
    # Generate data from parameters
    m_sim1 = initialize_model(TuringModels.karp_model5)
    Y = m_sim1.args.Y
    # Estimate parameters from data
    m_sim2 = TuringModels.karp_model5(Y, missing)
    mle_est_sim = optimize(m_sim2, MLE(), ConjugateGradient(),
                Optim.Options(iterations=10, store_trace=true, extended_trace=true)
    )
    @show mle_est_sim.optim_result

    println("\nRunning test of observed data MLE")
    # Load observed data and initialize model
    m_obs = initialize_model(TuringModels.karp_model5)
    # Estimate parameters from data
    Random.seed!(5)
    mle_est_obs = optimize(m_obs, MLE(), ConjugateGradient(),
                Optim.Options(iterations=10, store_trace=true, extended_trace=true)
    )
    @show mle_est_obs.optim_result
end

function test_functions()
    println("\nRunning test of initialize_model (setting: estimate model)")
    m = initialize_model(TuringModels.karp_model5)

    test_models()

    println("\nRunning test of multistart_MLE on initialized karp_model5")
    df = multistart_MLE(m, 2; maxiter=2)
    @show df

    println("\nRunning test of multistart_MLE on initialized karp_model5")
    res = estimate_MLE(TuringModels.karp_model5,  maxiter=3, Nseeds=3)
    @show res
    return
end
@time test_functions()

function test_turing_optimization()
    ###############################################
    # Test the optimization function and output
    m_obs = initialize_model(TuringModels.karp_model5)
    m_sim = initialize_model(TuringModels.karp_model5)
    Random.seed!(5)
    mle_estimate = optimize(m1, MLE(), ConjugateGradient(),
                Optim.Options(iterations=100_000, store_trace=true, extended_trace=true)
    )
    coeftable(mle_estimate) #! does not work: DomainError with -0.0006059430277771548:

    ##############################################
    # coeftable doesn't work -- investigate
    # Manually construct the VCOV matrix from the hessian
    θmle = convert_params_to_namedtuple(mle_estimate.values)
    LL(θ) = loglikelihood(m1, convert_params_to_namedtuple(θ))
    # Calculate the finite difference numerical hessian at the MLE location
    H1 = FiniteDiff.finite_difference_hessian(LL, mle_estimate.values.array)
    @show diag(H1)
    # Calculate the standard errors from the inverse of the Hessian
    VCOV1 = inv(-H1)
    se1 = sqrt.(diag(VCOV1))  # DomainError with -0.0006059426285795939, same error as coeftable
    #! there's a problem with the Hessian, it's not positive definite
    #! because it's concave up in σa² at the MLE (LL is increasing in decreasing σa²)
    #! Check what the MAP result would give

    ##############################################
    # Test the MAP optimization function and output
    Random.seed!(5)
    map_estimate = optimize(m1, MAP(), ConjugateGradient(),
                Optim.Options(iterations=100_000, store_trace=true, extended_trace=true)
    )
    coeftable(map_estimate) 
    #! MAP gives σa² estimate away from zero, so Hessian is positive definite (works)
    #! But LL is smaller at MAP than MLE above -- is the MLE biased or is this a problem with the model?
    #! Need to check simulation results to know what parameters give rise to MLE with hessian that is not positive definite
    LP(θ) = logjoint(m1, convert_params_to_namedtuple(θ))
    LP(map_estimate.values)
    # Calculate the finite difference numerical hessian at the MLE location
    Hmap = FiniteDiff.finite_difference_hessian(LP, map_estimate.values.array)
    @show diag(Hmap)
    VCOVmap = inv(-Hmap)
    semap = sqrt.(diag(VCOVmap))

    # Standard errors are calculated from the Fisher information matrix (inverse Hessian of the log likelihood or log joint)
    # Calculate the Hessian
    # Need the loglikelihood function as a function of one vector of parameters
    θmle = convert_params_to_namedtuple(mle_estimate.values)
    logjoint(m1, θmle)
    loglikelihood(m1, θmle)
    LJ(θ) = logjoint(m1, convert_params_to_namedtuple(θ))
    LL(θ) = loglikelihood(m1, convert_params_to_namedtuple(θ))
    # Calculate the finite difference numerical hessian at the MLE location
    H0 = FiniteDiff.finite_difference_hessian(LJ, mle_estimate.values.array; absstep=1e-10)
    H1 = FiniteDiff.finite_difference_hessian(LL, mle_estimate.values.array)
    VCOV0 = inv(-H0)
    VCOV1 = inv(-H1)
    se0 = sqrt.(diag(VCOV0))
    se1 = sqrt.(diag(VCOV1))  # DomainError with -0.0006059426285795939, same error as coeftable

    diag(H1)

    se = sqrt.(diag(
        inv(-FiniteDiff.finite_difference_hessian(LL, mle_estimate.values.array))
    ))
    # Try evaluating at exactly σa = 0
    sqrt.(diag(
        inv(-FiniteDiff.finite_difference_hessian(
            LL, 
            [0; mle_estimate.values.array[2:end]]
        ))
    ))
    diag(FiniteDiff.finite_difference_hessian(LL, [-1e-2; mle_estimate.values.array[2:end]]))
    FiniteDiff.finite_difference_jacobian(LL, [0; mle_estimate.values.array[2:end]])
    FiniteDiff.finite_difference_gradient(LL, [0; mle_estimate.values.array[2:end]])

    #! domain error here -- probably because of the σa parameter being close to 0.
    #! try again with σa being larger
    θmle2 = mle_estimate.values.array+[0.00013, 0, 0, 0, 0, 0, 0, 0, 0];
    H2 = FiniteDiff.finite_difference_hessian(LL, θmle2)
    VCOV2 = inv(-H2);
    se2 = sqrt.(diag(VCOV2))


    #? Try LL using flat priors
    result_freeρ_uniform4 = estimate_MLE(karp_model4; ρfixed=false, datatype="real", maxiter=100_000, Nseeds=20,
        σα²dist=Uniform(0, 1e10),
        σμ²dist=Uniform(0, 1e10),
        ρdist=Uniform(-1, 1),
        b₀sd=20, β₁sd=5, β₂sd=1
    )
    result_freeρ_uniform4.best_optim
    result_freeρ_uniform4.best_optim.values
    result_freeρ_uniform4.best_run.LL
    coeftable(result_freeρ_uniform4.best_optim)


    #? Try LL using flat priors and estimate v0
    result_freeρ_uniform5 = estimate_MLE(karp_model4; ρfixed=false, datatype="real", maxiter=100_000, Nseeds=20,
        σα²dist=Uniform(0, 1e10),
        σμ²dist=Uniform(0, 1e10),
        ρdist=Uniform(-1, 1),
        b₀sd=20, β₁sd=5, β₂sd=1,
        v0=missing
    )
    result_freeρ_uniform5.best_optim.values
    result_freeρ_uniform5.best_run.LL
    result_freeρ_uniform6.best_run.LL
    #? flatter prior on v0 did not change the result (same down to 1e-7)
    #? Which makes sense because the LL doesn't use the prior. The prior is only used for the MAP estimate.
    result_freeρ_uniform6.best_optim.values - result_freeρ_uniform5.best_optim.values

    coeftable(result_freeρ_uniform5.best_optim)
    diag(FiniteDiff.finite_difference_hessian(
        θ -> loglikelihood(
            result_freeρ_uniform5.best_optim.f.model, 
            convert_params_to_namedtuple(θ)
        ),
        result_freeρ_uniform5.best_optim.values.array
    ))


end # test_turing_optimization





######################################################################################
#    Test MLE on simulated data over range of parameters and simulated datasets
######################################################################################
# Setup parameter sampler from prior distribution
# Sample parameters from prior and simulate data from model
full_sampler = TuringModels.karp_model5(missing, missing)
full_sampler()

# Sample parameters from prior
param_sampler = TuringModels.karp_model5_parameters(missing;
    σα²dist=Exponential(1),
    σμ²dist=Exponential(1)
)
_θ = param_sampler()

# Sample parameters from prior given σα², σμ²
param_sampler2(σα², σμ²) = TuringModels.karp_model5_parameters(
    (; σα², σμ²);
    σα²dist=Exponential(1),
    σμ²dist=Exponential(1)
)
_θ = param_sampler2(1,1)()

# Simulate data from model given true parameters
data_sampler(θ) = TuringModels.karp_model5(missing, θ)
data_sampler(_θ)()

# Simulation loop settings
S = (;
    Nsigma = 2,  # length(σα² array); Nsigma^2 = number of σα², σμ² grid points
    Nparam = 2,  # Number of true parameter samples, conditional on σα², σμ²
    Nsim = 2,    # Number of simulated datasets per parameter sample
    Nseed = 2,   # Number of multistart seeds per simulated dataset to find MLE
)

# Create 2D grid for σα², σμ² simulations - log scale so many more points near 0
_x = range(-10, 10, length=S.Nsigma)
xx = exp10.(_x)
XY = [exp10.((σα², σμ²)) for σα² in _x, σμ² in _x ]
# For each σα², σμ² grid point, generate Nparam sets of parameters - b₀, β₁, β₂, ρ, v0
θmat = [param_sampler2(σα²σμ²...)() for σα²σμ² in XY, _ in 1:S.Nparam]

# Test MLE Estimate
# _res = estimate_MLE(TuringModels.karp_model5, S.Nsim; θ = θmat[1,1,1], maxiter=1_000, Nseeds=S.Nseed)
# @show _res.df_summary

# Define NamedTuple of matricies to fill in with results
results_names = [:mean_se, :mean_bias, :mean_abs_bias, :var_bias, :var_abs_bias, :Truth]
_mat() = deepcopy(zeros(size(θmat)))
_nt() = NamedTuple(n => _mat() for n in results_names)
results_nt = (; σα² = _nt(), σμ² = _nt())

#! Save intermittent results to file, load/update/save file - can I pickle the 3d array?
# about 7min per estimate_MLE with Nseeds=50
#! Want to know matrix of % of runs for that param vector that result in small σa^2 est. (e.g. < 0.1)
#? Can I just calc this after?


#! Check if parallelizing is maxed out (should be if Nseed is large enough)
#! Run on server for 1 day = 5 for Nsgima and Nparam
#! Run on server for 1 week = 11 for Nsgima and Nparam
#! Make distributed and run in Savio
total = S.Nsigma^2 * S.Nparam
counter = 0
for i in 1:S.Nsigma, j in 1:S.Nsigma, k in 1:S.Nparam
    perc_complete = round(counter/total*100, digits=2)
    @show i,j,k, perc_complete
    θ = θmat[i, j, k]
    # For each parameter vector, generate Nsim datasets
    res = estimate_MLE(TuringModels.karp_model5, S.Nsim; θ, maxiter=100_000, Nseeds=S.Nseed)
    @show res.df_summary
    # Store results in matrix
    for p in [:σα², :σμ²], n in results_names
        results_nt[p][n][i,j,k] = @subset(res.df_summary, res.df_summary.Parameter .== Symbol("θ.$p"))[1, n]
    end
    counter += 1
end
# Save results to file
save("tmp-results_nt.jld", "results_nt", results_nt)
# Load results from file
results_nt2 = load("tmp-results_nt.jld", "results_nt")

@show results_nt
m_ = mean(results_nt.σα².mean_se, dims=3)[:,:,1]
heatmap(xx, xx, mean(results_nt.σα².mean_se, dims=3)[:,:,1]; xscale=:log10, yscale=:log10, title="Mean σα² Standard Error")
heatmap(xx, xx, mean(results_nt.σα².mean_bias, dims=3)[:,:,1]; xscale=:log10, yscale=:log10, title="Mean σα² Bias")
heatmap(xx, xx, mean(results_nt.σα².mean_abs_bias, dims=3)[:,:,1]; xscale=:log10, yscale=:log10, title="Mean σα² Absolute Bias")

heatmap(1:2, 1:2, [1 2; 3 4])
# Save best runs
# Examine any for small simga_a^2 est. Maybe just sort them and see. Perhaps look at smallest sigma_a^2est / sigma_a^2












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

