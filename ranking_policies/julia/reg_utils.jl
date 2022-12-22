
using DataFrames
using StatsModels  # formulas and modelmatrix
using LinearAlgebra  # Diagonal




"""
mygls(formula, df, W, N)

@param formula: StatsModels formula for the linear regression model
@param df: dataframe with columns corresponding to the variables in formula
@param W: nxn weighting matrix, n = length(df)
@param N: number of units
@returns :β, :βse, :yhat, :resid, :βvar, :HC0, :sandwich

# References
- Bruce Hansen Econometrics (2021) section 17.15 Feasible GLS
- Greene Ed 7
- rsusmel lecture notes: bauer.uh.edu/rsusmel/phd/ec1-11.pdf
"""
function mygls(formula, df, W, N)
    println("Starting GLS")
    @info "mygls()" names(df) sizeW=size(W)
    X = StatsModels.modelmatrix(formula.rhs, df)
    y = StatsModels.modelmatrix(formula.lhs, df)
    XX = inv(X'X)
    XWX = X' * inv(W) * X
    NT, k = size(X)

    β = inv(XWX) * (X' * inv(W) * y)
    yhat = X * β
    resid = y - yhat
    XeX = X' * Diagonal(vec(resid .^ 2)) * X

    # Variance estimators
    # Hansen eq 4.16 (sandwich estimator)
    s = XX * XWX * XX
    s_root = diag(s) .^ 0.5
    # White estimator (rsusmel pg 11)
    HC0 = XX * XeX * XX
    HC0_root = diag(HC0) .^ 0.5
    # Greene (9-13)
    βvar = inv(XWX) / (NT - N - k)
    βse = diag(βvar) .^ 0.5

    # TODO: Add Newey-West HAC SE estimator.
    #   pg. 20 of bauer.uh.edu/rsusmel/phd/ec1-11.pdf

    results = Dict(:β => β,
        :βse => βse,
        :yhat => vec(yhat),
        :resid => resid,
        :βvar => βvar,
        :HC0 => HC0_root,
        :sandwich => s_root
    )
    # display(results)

    # Go with βse for now
    return(results)
end



"""
mygls_aggregate(formula, df, W, N)

@param formula: StatsModels formula for the linear regression model
@param df: dataframe with columns corresponding to the variables in formula
@param W: nxn weighting matrix, n = length(df)
@returns :β, :βse, :yhat, :resid, :βvar, :HC0, :sandwich

# References
- Bruce Hansen Econometrics (2021) section 17.15 Feasible GLS
- Greene Ed 7
- rsusmel lecture notes: bauer.uh.edu/rsusmel/phd/ec1-11.pdf
"""
function mygls_aggregate(formula, df, ρ; verbose=false)
    verbose ? println("Starting GLS") : nothing
    X = StatsModels.modelmatrix(formula.rhs, df)
    y = StatsModels.modelmatrix(formula.lhs, df)
    n = length(y)
    W = [ρ^abs(i-j) for i ∈ 1:n, j ∈ 1:n]
    Winv = inv(W)
    XX = inv(X'X)
    XWinvX = X' * Winv * X
    N, k = size(X)

    β = inv(XWinvX) * (X' * Winv * y)
    yhat = X * β
    v = y - yhat

    # Estimate σ² of ε (Greene Ed 7, eq (20-27))
    σ²v = (v'*Winv*v)[1] / N
    σ²θ = (1-ρ^2)*σ²v
    σ²θvar = 2*σ²θ^2 / N
    σ²θse = σ²θvar ^0.5
    βvar = σ²v*inv(XWinvX)
    βse = diag(βvar) .^0.5

    results = Dict(
        :β => β,
        :βse => βse,
        :yhat => vec(yhat),
        # :resid => v,
        :σ²v => σ²v,
        :σ²θ => σ²θ,
        :σ²θse => σ²θse
    )
    # display(results)

    return(results)
end
