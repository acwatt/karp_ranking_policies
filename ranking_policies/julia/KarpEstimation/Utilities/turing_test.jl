
println(@__DIR__)
using Pkg; Pkg.activate(joinpath(@__DIR__, "turing_test_env"))
# ]add Turing DataFrames LinearAlgebra Distributions CategoricalArrays Random Optim StatsBase StatsPlots ProgressMeter
using Turing
using DataFrames
using LinearAlgebra
using Optim
using StatsBase  # for coeftable and stderror
using StatsPlots  # for histogram
using ProgressMeter

module Model
    include("../Model/Model.jl")
end



# Generate Simulated data without trends or region fixed effects
N = 4; T = 60  # 1945-2005
ρ = 0.8785  # from r-scripts/reproducting_andy.R  -> line 70 result2
σ²base = 3.6288344  # σᵤ² from test_starting_real_estimation() after rescaling the data to units of 10,000 tons
θ = (ρ=ρ, σₐ²=σ²base, σᵤ²=σ²base)  # True paramter)
seed = 1234
data = Model.dgp(θ, N, T, seed)
histogram(data.eᵢₜ, label="eᵢₜ", title="Simulated data without trends or region fixed effects", legend=:topleft)

# Generate simulated data with trends
b₀ = [3.146, -0.454, 3.78, 3.479]
β = [0.265, 0]
true_params = (θ=θ, b₀=b₀, β=β)
data2 = Model.dgp(θ.ρ, θ.σₐ², θ.σᵤ², β, N, T; b₀ = b₀)
histogram(data2.eᵢₜ, label="eᵢₜ", title="Simulated data with trends", legend=:topleft)




θ₀ = (ρ=ρ, σₐ²=σ²base/2, σᵤ²=σ²base/2)  # Starting parameter in optimization search



#######################################################################
#                            Turing.jl Functions
#######################################################################


#######################################
#    Turing attempt for Karp model, no trends
#######################################
@model function karp_model_notrend(e,)

end


#######################################
#    Turing attempt for Karp model, with α and μ error terms
#######################################
function model1_many_params()

    @model function karp_model(eᵢₜ, eₜ)
        T = length(eₜ)
        N = length(eᵢₜ) ÷ T
        # Set variance priors
        σα² ~ InverseGamma(1, 1)
        σμ² ~ InverseGamma(1, 1)

        # Set error terms priors
        αₜ ~ MvNormal(zeros(T), sqrt(σα²)*I)
        μᵢₜ ~ MvNormal(zeros(N*T), sqrt(σμ²)*I)

        # Set FE and time trend priors
        b₀ ~ MvNormal(zeros(N), 2)
        B₀ = mean(b₀)
        β₁ ~ truncated(Normal(0, 0.5); lower=0)
        β₂ ~ Normal(0, 0.1)

        # Set AR(1) coefficient prior
        ρ ~ truncated(Normal(0.89, 0.05); lower=0, upper=1)

        # Initialize an empty vector to store the model AR(1) errors
        vₜ = Vector{Real}(undef, T)

        # DGP models
        # Period t = 1 period
        θ1 = αₜ[1] + sum(μᵢₜ[1:4])/N
        vₜ[1] = θ1  # Setting vₜ[0] = 0
        # Period Avg observation
        eₜ[1] ~ Normal(B₀ + β₁ + β₂, sqrt(σα² + σμ²/N))
        for i = 1:N
            # Period-unit observation
            eᵢₜ[i] ~ Normal(b₀[i] + β₁ + β₂, sqrt(σα² + σμ²))  
        end
        # Periods t = 2, ..., T
        for t = 2:T
            θₜ = αₜ[t] + sum(μᵢₜ[(4t-3):4t])/N
            vₜ[t] = ρ*vₜ[t-1] + θₜ
            # Period Avg observation
            eₜ[t] ~ Normal(B₀ + β₁*t + β₂*t^2 + ρ*vₜ[t-1], sqrt(σα² + σμ²/N))
            for i = 1:N
                # Period-unit observation
                eᵢₜ[i + (t-1)*N] ~ Normal(b₀[i] + β₁*t + β₂*t^2 + ρ*vₜ[t-1], sqrt(σα² + σμ²))
            end
        end

        return
    end

    eit = data2.eᵢₜ
    et = combine(groupby(data2, :t), :eᵢₜ => mean => :eₜ).eₜ

    # Estimate the MLE
    opt3 = optimize(karp_model(eit, et), MLE())
    opt3.values
    # Then, get the standard errors
    infomat = stderror(opt3)
    coefdf = coeftable(opt3)

end



#######################################
#    Turing attempt for Karp model, without α and μ error terms
#######################################
@model function karp_model2(eᵢₜ, eₜ; ρ=missing)
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

    # DGP models
    for t = 1:T
        # AR(1) error variances
        σₜ² = (σα² + σμ²/N)*sum(ρ^(2*(s-1)) for s=1:t)
        σᵢₜ² = (σα² + σμ²)*sum(ρ^(2*(s-1)) for s=1:t)
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
opt4 = optimize(karp_model2(eit, et), MLE())
opt4.lp
opt4.values
opt4.optim_result
# Then, get the standard errors
coefdf = coeftable(opt4)

# Try again and fix ρ
# Estimate the MLE
df = DataFrame(LL=Float64[], iters=Float64[], σα²=Float64[], σμ²=Float64[], ρ=Float64[],
               iconverge=Bool[], gconverge=Bool[], fconverge=Bool[], gtol=Float64[])
@showprogress for _ in 1:10
opt5 = optimize(
    karp_model2(eit, et; ρ=ρ), MLE(), 
    ConjugateGradient(),
    Optim.Options(iterations=30_000, g_tol = 1e-12, store_trace = true, show_trace=false)
);
@show push!(df, (opt5.lp,  # Log likelihood value
                 opt5.optim_result.iterations,  # number of iterations
                 opt5.values[1], opt5.values[2], ρ,  # estimated parameters
                 opt5.optim_result.iteration_converged,  # did it hit max iterations?
                 opt5.optim_result.g_converged, # did it converge from the gradient?
                 opt5.optim_result.f_converged,  # did it converge from the LL value?
                 opt5.optim_result.g_abstol,  # gradient tolerance set
                 ))
@show opt5.optim_result.stopped_by
end



############################## try with many periods
# Generate simulated data with many time periods
T3 = 200
data3 = Model.dgp(θ.ρ, θ.σₐ², θ.σᵤ², β, N, T3; b₀ = b₀)
histogram(data3.eᵢₜ, label="eᵢₜ", title="Simulated data with trends", legend=:topleft)

df3 = DataFrame(LL=Float64[], iters=Float64[], σα²=Float64[], σμ²=Float64[], ρ=Float64[],
               iconverge=Bool[], gconverge=Bool[], fconverge=Bool[], gtol=Float64[], N=Int64[], T=Int64[])
optim_list = []
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
    push!(optim_list, opt6)
end
coeftable(opt6)






#######################################
#  Turing attempt for Karp model, adding time averaging
#######################################
#! Add time averaging to this model
@model function karp_model3(eᵢₜ, eₜ; ρ=nothing)
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

