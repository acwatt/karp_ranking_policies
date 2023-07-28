
println(@__DIR__)
using Pkg; Pkg.activate(joinpath(@__DIR__, "turing_test_env"))
# ]add Turing DataFrames LinearAlgebra Distributions CategoricalArrays Random Optim StatsBase StatsPlots ProgressMeter DataFramesMeta Dates CSV Statistics
using Turing
using DataFrames, DataFramesMeta
using LinearAlgebra
using Optim
using StatsBase  # for coeftable and stderror
using StatsPlots  # for histogram
using ProgressMeter
using Random
using Parameters

module Model
    include("../Model/Model.jl")
end
include("HelperFunctions.jl")  # HF






#######################################################################
#                            Turing.jl Functions
#######################################################################


#######################################
#    Turing attempt for Karp model, no trends
#######################################
#! try estimating a model without trends on simulated data
function model1_no_trends()
    N = 4; T = 60  # 1945-2005
    ρ = 0.8785  # from r-scripts/reproducting_andy.R  -> line 70 result2
    σ²base = 3.6288344  # σᵤ² from test_starting_real_estimation() after rescaling the data to units of 10,000 tons
    θ = (ρ=ρ, σₐ²=σ²base, σᵤ²=σ²base)  # True paramter)
    seed = 1234
    data = Model.dgp(θ, N, T, seed)

    @model function karp_model_notrend(e,)

    end
end





#######################################
#    Turing attempt for Karp model, without α and μ error terms
#######################################
function model2_few_params()
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

#######################################
#  Model with v error vector to hold values
#######################################
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

#! Add the fixed ρ version to estimate_model()
# Estimate the MLE
opt6 = optimize(karp_model3(eit, et), MLE())
# Estimate the MLE, fixed ρ
opt7 = optimize(karp_model3(eit, et, ρ=θ.ρ), MLE())

function estimate_model(model; datatype="real", maxiter=10_000, seeds=100)
    # Estimate the MLE on real data
    if datatype == "real"
        # load the data, then transform eit to eit/1000
        data = @chain HF.read_data(N, T) @transform(:eᵢₜ = :eᵢₜ ./ 1000)
    elseif datatype == "simulated"
        # simulate data based on approximate parameters recovered from data
        b₀ = [3.146, -0.454, 3.78, 3.479]
        β = [0.265, 0]
        data = Model.dgp(θ.ρ, θ.σₐ², θ.σᵤ², β, N, T; b₀ = b₀)
    end
    eit = data.eᵢₜ
    et = combine(groupby(data, :t), :eᵢₜ => mean => :eₜ).eₜ

    # Initialize dataframe to store the results
    optim_df = DataFrame(UID=Int64[], LL=Float64[], iters=Float64[], 
                    σα²=Float64[], σμ²=Float64[], ρ=Float64[],
                    b01=Float64[], b02=Float64[], b03=Float64[], b04=Float64[],
                    β1=Float64[], β2=Float64[],
                iconverge=Bool[], gconverge=Bool[], fconverge=Bool[], gtol=Float64[],
                N=Int64[], T=Int64[], ρfixed=Bool[]
    )
    optim_dict = Dict()
    UID = 1

    # run optimize on the model 100 times, adding the results to a dataframe
    @showprogress for seed in 1:seeds
        Random.seed!(seed)
        opt = optimize(model(eit, et),
                        MLE(), ConjugateGradient(),
                        Optim.Options(iterations=maxiter))
        @show push!(optim_df, (
            UID, opt.lp,  # Log likelihood value
            opt.optim_result.iterations,  # number of iterations
            opt.values[:σα²], opt.values[:σμ²], opt.values[:ρ],
            opt.values[Symbol("b₀[1]")], opt.values[Symbol("b₀[2]")], opt.values[Symbol("b₀[3]")], opt.values[Symbol("b₀[4]")],
            opt.values[:β₁], opt.values[:β₂],
            opt.optim_result.iteration_converged,  # did it hit max iterations?
            opt.optim_result.g_converged, # did it converge from the gradient?
            opt.optim_result.f_converged,  # did it converge from the LL value?
            opt.optim_result.g_abstol,  # gradient tolerance setting
            length(eit) ÷ length(et), length(et), false,  # N, T, ρ is fixed
        ))
        optim_dict[UID] = opt
        UID += 1
    end

    # Find best run (highest LL), and return that result
    best_run = optim_df[findmax(optim_df.LL)[2], :]
    best_optim = optim_dict[best_run.UID]
    dict = Dict{String,Any}()
    @pack! dict = best_run, best_optim, optim_df, optim_dict
    return dict
end

res1 = estimate_model(karp_model3; maxiter=50_000)
res1["best_run"]
#! Would want to use coeftable but there's a negative in the vcov matrix
# 
mat = [res1["best_optim"].values sqrt.(abs.(diag(vcov(res1["best_optim"]))))]
# Create 95% confidence intervals
LB = mat[:, 1] .- 1.96*mat[:, 2]
UB = mat[:, 1] .+ 1.96*mat[:, 2]
mat = [mat LB UB]


"""
Parameter estimates for data / 1000
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


Parameter estimates for data
9-element Named Vector{Float64}
A     │
──────┼──────────
σα²   │       0.0
σμ²   │ 3.99063e5
b₀[1] │   1230.16
b₀[2] │  -89.1669
b₀[3] │   1462.39
b₀[4] │   1351.87
β₁    │   90.3224
β₂    │ 0.0849059
ρ     │  0.929347

9-element Named Vector{Float64}
A     │
──────┼───────────
σα²   │    1.23397
σμ²   │    1.52001
b₀[1] │    1.10942
b₀[2] │    1.10942
b₀[3] │    1.10942
b₀[4] │    1.10942
β₁    │  0.0884016
β₂    │ 0.00147306
ρ     │  0.0489734
"""

















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
@model function karp_model4(eᵢₜ, eₜ; ρ=nothing)
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

