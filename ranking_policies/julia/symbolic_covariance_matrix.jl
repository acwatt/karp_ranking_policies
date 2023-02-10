using Symbolics
@variables ρ σₐ² σᵤ² n
N_ = 2; T_ = 2;
vv(i,j,s) = Evᵢₜvⱼₜ₊ₛ(ρ, σₐ², σᵤ², i, j, s, n)

V_ = [vv(1,1,0) vv(1,2,0) vv(1,1,1) vv(1,2,1)
    vv(2,1,0) vv(2,2,0) vv(2,1,1) vv(2,2,1)
    vv(1,1,-1) vv(1,2,-1) vv(1,1,0) vv(1,2,0)
    vv(2,1,-1) vv(2,2,-1) vv(2,1,0) vv(2,2,0)]

latexify(V_) |> print
s = latexify(V_^-1)

open("Vinv.txt","w") do io
    print(io, latexify(V_^-1))
end

simplify.(V_^-1)
latexify(simplify.(V_^-1)) |> print



function Σ(ρ::Num, σₐ²::Num, σᵤ²::Num, N, T; verbose = false)
    # Initalize matrix of 0s
    V = Array{Num}(undef,N*T,N*T)

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
        # println(a)
    end

    # Fill in lower triangle by symmetry
    V = Symmetric(V)
    if verbose
        latexify(V) |> print
    end
    return (V)
end






# Run Σ() from covariance_matrix.jl first
# Then see what the symbolic V is
Σ(ρ, σₐ², σᵤ², N_, T_; verbose = true);