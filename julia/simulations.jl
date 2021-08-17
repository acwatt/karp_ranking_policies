# simulations.jl
# author = Aaron Watt
# email = aaron@acwatt.net
# description = Estimations on simulated data to test estimation issues for project
# Notes:
#       - all nu's are the symbol \nu not the letter v

# packages
using LinearAlgebra, Statistics, Random, Distributions

# create simulated data
b = 1
n = 2
T = 3
tvec = [t for t in 1:T]
b₀ = [1, 1]
a = [1, 0.5]
ν₀ = 0
β = [1, 1]
ρ = 0.9
σₐ, σᵤ = 1, 1
α = rand(Normal(0, σₐ), T)
μ = rand(Normal(0, σᵤ), n, T)

function νt(t)
    if t == 0
        return ν₀
    else
        return 1/n*sum([νit(i, t) for i in 1:n])
    end
end

function νit(i, t)
    return ρ*νt(t-1) + α[t] + μ[i, t]
end

ν = νt.(tvec)

# for i in 1:n, create n rows for the n dummies: [1 0 t t^2]
# for t in 1:T, create another set of n rows for time t
X = [[t for t in 1:T]; for i in 1:n]
# need to build the matrix X of t and t^2
function xt(t)
    return []
end

X = xt.(tvec)

function eit(i, t)..
    1/b*(b₀ + dot(a, tvec))
end
