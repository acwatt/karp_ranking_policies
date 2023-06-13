#==============================================================================
file: Model.jl
description: defines model equations for ranking policies project
author: Aaron C Watt (UCB Grad Student, Ag & Resource Econ)
email: aaron@acwatt.net
created: 2023-06-13
last update: 
See docs/code/Model.md for more notes
==============================================================================#
module Model
#------------------- COVARIANCE MATRIX ---------------------
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


#------------------- SYMBOLIC COVARIANCE MATRIX ---------------------
# Moved to symbolic_covariance_matrix.jl






#------------------- COVARIANCE MATRIX DERIVATIVE ---------------------
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



end