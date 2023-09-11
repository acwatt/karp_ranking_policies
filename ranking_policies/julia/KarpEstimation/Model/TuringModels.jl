""" TuringModels module

This module contains the Turing models used in the Karp estimation project.
"""
module TuringModels
using Turing
using DataFrames, DataFramesMeta
using LinearAlgebra
using Optim
using Random
using Parameters
using FiniteDiff
using NamedArrays


"""Karp model, AR(1) variances"""
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


"""Karp model, AR(1) errors, informative priors"""
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


"""Karp model, AR(1) errors, flat priors"""
@model function karp_model4(eᵢₜ, eₜ; 
        σα²=missing, σμ²=missing, b₀=missing, β₁=missing, β₂=missing, ρ=missing, 
        σα²dist=InverseGamma(1, 1), 
        σμ²dist=InverseGamma(1, 1),
        ρdist=truncated(Normal(0.87, 0.05); lower=0, upper=1),
        b₀sd=2, β₁sd=0.5, β₂sd=0.1,
        v0 = 0
    )
    # Set function usage
    # If no parameters and data, we are sampling the parameters from the prior
    usage = if ismissing(σα²) && ismissing(eᵢₜ)
        "sample_params"
    elseif ismissing(eᵢₜ)  # If no data, we are sampling data
        "sample_data"
    else  # If data is given, estimate the model from data
        "estimate_model"
    end
    if (usage=="sample_data") | (usage=="sample_params")
        T=60; N=4
        eₜ = Vector{Real}(undef, T)
        eᵢₜ = Vector{Real}(undef, T*N)
    else
        T = length(eₜ)
        N = length(eᵢₜ) ÷ T
    end
    # If no parameters, we are sampling the prior or estimating the model
    if usage == "sample_params"
        # Set variance priors
        σα² ~ σα²dist
        σμ² ~ σμ²dist
        # Set FE and time trend priors
        b₀ ~ MvNormal(zeros(N), b₀sd)
        β₁ ~ truncated(Normal(0, β₁sd); lower=0)
        β₂ ~ Normal(0, β₂sd)
    end
    # if σα² is given, then we sample data below given parameters as arguments

    B₀ = mean(b₀)

    # Set AR(1) coefficient prior
    if ismissing(ρ)
        ρ ~ ρdist
    end

    # Initialize an empty vector to store the model AR(1) errors
    vₜ₋₁ = Vector{Real}(undef, T+1)
    # This has T+1 elements because we need to store initial value vₜ₋₁[1]
    if ismissing(v0)  # If no initial value, estimate or sample it
        vₜ₋₁[1] ~ Normal(0, 10)
    else
        vₜ₋₁[1] = v0
    end

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
    #! This is not working -- must separate into different functions
    #! use multiple dispatch
    if usage == "sample_params"
        return (; σα², σμ², b₀, β₁, β₂, ρ, v0=vₜ₋₁[1])
    elseif usage == "sample_data"
        return (; eᵢₜ, eₜ)
    else  # estimate the model from data
        return
    end
end

@model function karp_model5_parameters(θ;
    N=4,
    σα²dist=Uniform(0, 1e10),
    σμ²dist=Uniform(0, 1e10),
    ρdist=Uniform(0, 1),
    b₀sd=20, β₁sd=5, β₂sd=1,
    )
    # Initialize parameters
    params = (:σα², :σμ², :b₀, :β₁, :β₂, :ρ, :v0)
    if ismissing(θ)
        θ = NamedTuple{(:σα², :σμ², :b₀, :β₁, :β₂, :ρ, :v0)}(repeat([missing],7))
    else
        for k in params
            if !haskey(θ, k)
                θ = merge(θ, (;k => missing))
            end
        end
    end

    # Set variance priors
    θ.σα² ~ σα²dist
    θ.σμ² ~ σμ²dist
    # Set AR(1) coefficient prior
    θ.ρ ~ ρdist
    # Set FE and time trend priors
    θ.β₁ ~ truncated(Normal(0, β₁sd); lower=0)
    θ.β₂ ~ Normal(0, β₂sd)
    θ.b₀ ~ MvNormal(zeros(N), b₀sd)
    B₀ = mean(θ.b₀)
    # Set initial AR(1) error prior
    θ.v0 ~ Normal(0, 10)
    return θ
end

"""Karp model, AR(1) errors, flat priors, all parameters are estimated"""
@model function karp_model5(Y, θ;
    N=4, T=60, usage="estimate_model",
    σα²dist=Exponential(1),
    σμ²dist=Exponential(1),
    ρdist=Uniform(0, 1),
    b₀sd=20, β₁sd=5, β₂sd=1,
    )
    # INITIALIZE DATA
    if ismissing(Y)
        Y = (;
            eₜ = Vector{Real}(undef, T),
            eᵢₜ = Vector{Real}(undef, T*N)
        )
        # if data and θ are both missing, we must be sampling parameters from the prior
        usage = ismissing(θ) ? "sample_params" : usage
    else
        T = length(Y.eₜ)
        N = length(Y.eᵢₜ) ÷ T
    end

    # INITIALIZE PARAMETERS
    params = (:σα², :σμ², :b₀, :β₁, :β₂, :ρ, :v0)
    if ismissing(θ)
        θ = NamedTuple{(:σα², :σμ², :b₀, :β₁, :β₂, :ρ, :v0)}(repeat([missing],7))
    else
        for k in params
            if !haskey(θ, k)
                θ = merge(θ, (;k => missing))
            end
        end
    end

    # SET PRIORS
    # Set variance priors
    θ.σα² ~ σα²dist
    θ.σμ² ~ σμ²dist
    # Set AR(1) coefficient prior
    θ.ρ ~ ρdist
    # Set FE and time trend priors
    θ.β₁ ~ truncated(Normal(0, β₁sd); lower=0)
    θ.β₂ ~ Normal(0, β₂sd)
    θ.b₀ ~ MvNormal(zeros(N), b₀sd)
    B₀ = mean(θ.b₀)
    # Set initial AR(1) error prior
    θ.v0 ~ Normal(0, 5)

    # Initialize AR(1) errors, with T+1 elements because we need to store initial value vₜ₋₁[1]
    vₜ₋₁ = Vector{Real}(undef, T+1)
    vₜ₋₁[1] = θ.v0

    # DGP MODEL
    for t = 1:T
        # Period Avg observation
        Y.eₜ[t] ~ Normal(B₀ + θ.β₁*t + θ.β₂*t^2 + θ.ρ*vₜ₋₁[t], sqrt(θ.σα² + θ.σμ²/N))
        vₜ₋₁[t+1] = Y.eₜ[t] - B₀ - θ.β₁*t - θ.β₂*t^2
        for i = 1:N
            # Period-unit observation
            Y.eᵢₜ[(t-1)*N + i] ~ Normal(θ.b₀[i] + θ.β₁*t + θ.β₂*t^2 + θ.ρ*vₜ₋₁[t], sqrt(θ.σα² + θ.σμ²))
        end
    end

    return (; Y, θ)
end

"""Karp model, AR(1) errors, flat priors, all parameters are estimated"""
@model function karp_model6(Y, θ;
    N=4, T=60, usage="estimate_model",
    σα²dist=Exponential(1),
    σμ²dist=Exponential(1),
    ρdist=Uniform(0, 1),
    b₀sd=20, β₁sd=5, β₂sd=1,
    )
    # INITIALIZE DATA
    if ismissing(Y)
        Y = (;
            eₜ = Vector{Real}(undef, T),
            eᵢₜ = Vector{Real}(undef, T*N)
        )
        # if data and θ are both missing, we must be sampling parameters from the prior
        usage = ismissing(θ) ? "sample_params" : usage
    else
        T = length(Y.eₜ)
        N = length(Y.eᵢₜ) ÷ T
    end

    # INITIALIZE PARAMETERS
    params = (:σα², :σμ², :b₀, :β₁, :β₂, :ρ, :v0)
    if ismissing(θ)
        θ = NamedTuple{(:σα², :σμ², :b₀, :β₁, :β₂, :ρ, :v0)}(repeat([missing],7))
    else
        for k in params
            if !haskey(θ, k)
                θ = merge(θ, (;k => missing))
            end
        end
    end

    # SET PRIORS
    # Set variance priors
    θ.σα² ~ σα²dist
    θ.σμ² ~ σμ²dist
    # Set AR(1) coefficient prior
    θ.ρ ~ ρdist
    # Set FE and time trend priors
    θ.β₁ ~ truncated(Normal(0, β₁sd); lower=0)
    θ.β₂ ~ Normal(0, β₂sd)
    θ.b₀ ~ MvNormal(zeros(N), b₀sd)
    B₀ = mean(θ.b₀)
    # Set initial AR(1) error prior
    θ.v0 ~ Normal(0, 10)

    # Initialize AR(1) errors, with T+1 elements because we need to store initial value vₜ₋₁[1]
    vₜ₋₁ = Vector{Real}(undef, T+1)
    vₜ₋₁[1] = θ.v0

    # DGP MODEL
    for t = 1:T
        # Period Avg observation
        Y.eₜ[t] ~ Normal(B₀ + θ.β₁*t + θ.β₂*t^2 + θ.ρ*vₜ₋₁[t], sqrt(θ.σα² + θ.σμ²/N))
        vₜ₋₁[t+1] = Y.eₜ[t] - B₀ - θ.β₁*t - θ.β₂*t^2
        for i = 1:N
            # Period-unit observation
            Y.eᵢₜ[(t-1)*N + i] ~ Normal(θ.b₀[i] + θ.β₁*t + θ.β₂*t^2 + θ.ρ*vₜ₋₁[t], sqrt(θ.σα² + θ.σμ²))
        end
    end
    for i = 1:N
        RHSmean_i = mean([θ.b₀[i] + θ.β₁*t + θ.β₂*t^2 + θ.ρ*vₜ₋₁[t] for t=1:T])
        Y.eᵢ[i] ~ Normal(RHSmean_i, sqrt(θ.σα²/T))
    end
    # Impose restriction: mean(μᵢₜ)= 0 over all i,t
    # RHSmean = mean([θ.b₀[i] + θ.β₁*t + θ.β₂*t^2 + θ.ρ*vₜ₋₁[t] for t=1:T, i=1:N])
    # Y.e ~ Normal(RHSmean, sqrt(θ.σα²/T))

    return (; Y, θ)
end

end
#=
"""
# Initialize AR(1) errors, with T+1 elements because we need to store initial value vₜ₋₁[1]
vₜ₋₁ = Vector{Real}(undef, T+1)
vₜ₋₁[1] = v0
B₀ = mean(b₀)

# DGP MODEL
for t = 1:T
    # Period Avg observation
    eₜ[t] ~ Normal(B₀ + β₁*t + β₂*t^2 + ρ*vₜ₋₁[t], sqrt(σα² + σμ²/N))
    vₜ₋₁[t+1] = eₜ[t] - B₀ - β₁*t - β₂*t^2
    for i = 1:N
        # Period-unit observation
        eᵢₜ[(t-1)*N + i] ~ Normal(b₀[i] + β₁*t + β₂*t^2 + ρ*vₜ₋₁[t], sqrt(σα² + σμ²))
    end
end
"""
=#