#==============================================================================
file: Model.jl
description: defines model equations for ranking policies project
author: Aaron C Watt (UCB Grad Student, Ag & Resource Econ)
email: aaron@acwatt.net
created: 2023-06-13
last update: 
See docs/code/Model.md for more notes
==============================================================================#
# module Model  # removing module so functions can be directly called in EstimatorsMLE
using Random
using Distributions
using CategoricalArrays
using DataFrames


##################### COVARIANCE MATRIX #####################
# Building the covariance matrix for the N=2, T=3 case
# Not needed anymore but nice to keep for tests
N = 2;
T = 3;
A(ρ, σₐ², σᵤ²) = 1 / (1 - ρ^2) * σₐ² + (1 + ρ^2 / (N * (1 - ρ^2))) * σᵤ²;
B(ρ, σₐ², σᵤ²) = 1 / (1 - ρ^2) * σₐ² + (ρ^2 / (N * (1 - ρ^2))) * σᵤ²;
C(ρ, σₐ², σᵤ²) = ρ / (1 - ρ^2) * σₐ² + (ρ / N + ρ^3 / (N * (1 - ρ^2))) * σᵤ²;
E(ρ, σₐ², σᵤ²) = ρ * C(ρ, σₐ², σᵤ²);

function Σ(ρ, σₐ², σᵤ²)
    [A(ρ, σₐ², σᵤ²) B(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²) E(ρ, σₐ², σᵤ²) E(ρ, σₐ², σᵤ²)
        B(ρ, σₐ², σᵤ²) A(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²) E(ρ, σₐ², σᵤ²) E(ρ, σₐ², σᵤ²)
        C(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²) A(ρ, σₐ², σᵤ²) B(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²)
        C(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²) B(ρ, σₐ², σᵤ²) A(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²)
        E(ρ, σₐ², σᵤ²) E(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²) A(ρ, σₐ², σᵤ²) B(ρ, σₐ², σᵤ²)
        E(ρ, σₐ², σᵤ²) E(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²) C(ρ, σₐ², σᵤ²) B(ρ, σₐ², σᵤ²) A(ρ, σₐ², σᵤ²)]
end


# Definitions from Eq. (2), (3), (4)
# Building the covariance matrix for general N, T
# Indicator functions
"""eq (2) of 2021-12 writeup"""
ι(i, j) = (i == j ? 1 : 0)
"""eq (2) of 2021-12 writeup"""
κ(s) = (s == 0 ? 0 : 1)
"""eq (3) of 2021-12 writeup"""
χ(ρ, i, j, s, N) = (1 - κ(s)) * ι(i, j) + κ(s) * ρ^s / N + ρ^(2 + s) / (N * (1 - ρ^2))  # realized the last ρ wasn't squared (2023-01-16)


"""Element of the covariance matrix: Σ = E[vᵢₜvⱼₜ₊ₛ] - eq (4) of 2021-12 writeup"""
Evᵢₜvⱼₜ₊ₛ(ρ, σₐ², σᵤ², i, j, s, N; b = 1) = 1 / b^2 * (σₐ² * ρ^s / (1 - ρ^2) + χ(ρ, i, j, s, N) * σᵤ²)


##################### SYMBOLIC COVARIANCE MATRIX #####################
# Moved to symbolic_covariance_matrix.jl






##################### COVARIANCE MATRIX DERIVATIVE #####################
"""i,t, j,t+s Elements of ∂Σ/∂y for y ∈ {ρ, σₐ², σᵤ²} - Eq (8) of 2021-12 writeup"""
∂Σ∂ρ(ρ, σₐ², σᵤ², i, j, s, N) = ρ^(s-1) / N / (ρ^2 - 1)^2 * (
    (s*κ(s) + (2 + s - 2*s*κ(s))*ρ^2 + (s*κ(s) - s)*ρ^4)*σᵤ² +
    ((2-s)*N*ρ^2 + N*s)*σₐ²
)

"""Eq (9) of 2021-12 writeup"""
∂Σ∂σₐ²(ρ, σₐ², σᵤ², i, j, s, N) = ρ^s / (1 - ρ^2)
"""Eq (9) of 2021-12 writeup"""
∂Σ∂σᵤ²(ρ, σₐ², σᵤ², i, j, s, N) = χ(ρ, i, j, s, N)

derivatives = Dict("ρ" => ∂Σ∂ρ, "σₐ²" => ∂Σ∂σₐ², "σᵤ²" => ∂Σ∂σᵤ²)

# General derivative ∂Σ/∂y for y ∈ {"ρ", "σₐ²", "σᵤ²"} (quotes needed for y input)
∂Σ∂y(y, ρ, σₐ², σᵤ², i, j, s, N) = derivatives[y](ρ, σₐ², σᵤ², i, j, s, N)


"""
    Σ(ρ, σₐ², σᵤ², N, T)

Return the residuals' analytical N*TxN*T covariance matrix given values of parameters
@param ρ: float in [0,1], decay rate of AR1 emissions process
@param σₐ²: float >0, SD of year-specific emissions shock 
@param σᵤ²: float >0, SD of region-year-specific emissions shock
@param N: int>1, number of units/regions
@param T: int>1, number of time periods
"""
function Σ(ρ::Real, σₐ²::Real, σᵤ²::Real, N, T; verbose = false)
    # Initalize matrix of 0s
    V = zeros(N * T, N * T)

    # Fill in upper triangle
    idx = [(i, j) for i ∈ 1:N*T for j ∈ i:N*T]
    for (row, col) in idx
        t = Integer(ceil(row / N))
        i = row - (t - 1) * N

        τ = Integer(ceil(col / N))
        s = τ - t
        j = col - (τ - 1) * N

        V[row, col] = Evᵢₜvⱼₜ₊ₛ(ρ, σₐ², σᵤ², i, j, s, N)
        a = [i, t, j, τ, row, col]
    end

    # Fill in lower triangle by symmetry
    V = Symmetric(V)
    if verbose
        println("\nFull Covariance Matirx:")
        for r in 1:size(V)[1]
            println(round.(V[r, :]; sigdigits = 4))
        end
    end
    return (V)
end


##################### DATA GENERATING PROCESS #####################

"""
    dgp(ρ, σₐ², σᵤ², β, N, T; v₀, μ₀, σ₀)

Return simulated data from the data generating process given paramters.

# Arguments
- `ρ::Float64`: decay rate of AR1 emissions process ∈ [0,1]
- `σₐ²::Float64`: SD of year-specific emissions shock , >0
- `σᵤ²::Float64`: SD of region-year-specific emissions shock, >0
- `β::Array{Float64,2}`: linear and quadratic time trend parameters
- `v₀::Float64`: initial emissions shock of AR1 process (v_t where t=0)
- `b₀::Array{Float64,N}`: b₀ᵢ for different regions i in {1, ..., N}
    if not given, then μ₀ and σ₀ are used to pull b₀ᵢ from a random distribution
- `μ₀::Float64`: mean of region-specific fixed effect distribution (b₀ᵢ)
- `σ₀::Float64`: SD of region-specific fixed effect distribution (b₀ᵢ), >0
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


# end