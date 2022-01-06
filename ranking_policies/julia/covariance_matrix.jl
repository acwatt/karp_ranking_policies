# Loading packages can often take a long time, but Julia is very fast once packages are loaded
# using Zygote  # for reverse-mode auto-differentiation
using Random  # Random numbers
using Distributions
using DataFrames
using LinearAlgebra  # Diagonal
using StatsModels  # formulas and modelmatrix
using GLM  # OLS (lm) and GLS (glm)
using Optim  # optimize (minimizing)
# using Symbolics  # symbolic functions
import Dates
using Latexify  # output symbolic expressions as latex code
# using ModelingToolkit  # more symbolic matrix functions
# using Reduce  # more symbolic matrix functions (specifically Reduce.Algebra.^)
# @force using Reduce.Algebra  # to extend native functions to Sybmol/Expr types
using Logging
import CSV

#######################################################################
#           Setup
#######################################################################
io = open("log.txt", "a")
logger = SimpleLogger(io)
# debuglogger = ConsoleLogger(stderr, Logging.Debug)
global_logger(logger)
s = "\n"^20 * "="^60 * "\nCOVARIANCE MATRIX ESTIMATION BEGIN\n" * "="^60
@info s Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")




#######################################################################
#           Functions
#######################################################################

# Building the covariance matrix
N = 2;
T = 3;
A(ρ, σₐ, σᵤ) = 1 / (1 - ρ^2) * σₐ^2 + (1 + ρ^2 / (N * (1 - ρ^2))) * σᵤ^2;
B(ρ, σₐ, σᵤ) = 1 / (1 - ρ^2) * σₐ^2 + (ρ^2 / (N * (1 - ρ^2))) * σᵤ^2;
C(ρ, σₐ, σᵤ) = ρ / (1 - ρ^2) * σₐ^2 + (ρ / N + ρ^3 / (N * (1 - ρ^2))) * σᵤ^2;
E(ρ, σₐ, σᵤ) = ρ * C(ρ, σₐ, σᵤ);

function Σ(ρ, σₐ, σᵤ)
    [A(ρ, σₐ, σᵤ) B(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) E(ρ, σₐ, σᵤ) E(ρ, σₐ, σᵤ)
        B(ρ, σₐ, σᵤ) A(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) E(ρ, σₐ, σᵤ) E(ρ, σₐ, σᵤ)
        C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) A(ρ, σₐ, σᵤ) B(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ)
        C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) B(ρ, σₐ, σᵤ) A(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ)
        E(ρ, σₐ, σᵤ) E(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) A(ρ, σₐ, σᵤ) B(ρ, σₐ, σᵤ)
        E(ρ, σₐ, σᵤ) E(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) B(ρ, σₐ, σᵤ) A(ρ, σₐ, σᵤ)]
end




# Indicator functions
ι(i, j) = (i == j ? 1 : 0)
κ(s) = (s == 0 ? 0 : 1)
χ(ρ, i, j, s, N) = (1 - κ(s)) * ι(i, j) + κ(s) * ρ^s / N + ρ^(2 + s) / (N * (1 - ρ))
# Symbolic function
χ(i, j, s, N) = (1 - κ(s)) * ι(i, j) + κ(s) * ρ^s / N + ρ^(2 + s) / (N * (1 - ρ))

# Element of the covariance matrix: E[vᵢₜvⱼₜ₊ₛ]
Evᵢₜvⱼₜ₊ₛ(ρ, σₐ, σᵤ, i, j, s, N; b = 1) = 1 / b^2 * (σₐ^2 * ρ^s / (1 - ρ^2) + χ(ρ, i, j, s, N) * σᵤ^2)
# Symbolic function
Evᵢₜvⱼₜ₊ₛ(i, j, s, N; b = 1) = 1 / b^2 * (σₐ^2 * ρ^s / (1 - ρ^2) + χ(i, j, s, N) * σᵤ^2)


"""
    Σ(ρ, σₐ, σᵤ, N, T)

Return the residuals' analytical N*TxN*T covariance matrix given values of parameters
@param ρ: float in [0,1], decay rate of AR1 emissions process
@param σₐ: float >0, SD of year-specific emissions shock 
@param σᵤ: float >0, SD of region-year-specific emissions shock
@param N: int>1, number of units/regions
@param T: int>1, number of time periods
"""
function Σ(ρ, σₐ, σᵤ, N, T; verbose = false)
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

        V[row, col] = Evᵢₜvⱼₜ₊ₛ(ρ, σₐ, σᵤ, i, j, s, N)
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

"""
Symbolic function
@param verbose: bool, if true, print the covariance matrix elements out
"""
function Σ(N, T; verbose = false)
    # Initalize matrix of 0s
    V = Array{Num}(undef, N * T, N * T)
    verbose ? println("\nELEMENTS OF THE COVARIANCE MATRIX:") : nothing

    # Fill in upper triangle
    idx = [(i, j) for i ∈ 1:N*T for j ∈ i:N*T]
    for (row, col) in idx
        t = Integer(ceil(row / N))
        i = row - (t - 1) * N

        τ = Integer(ceil(col / N))
        s = τ - t
        j = col - (τ - 1) * N

        V[row, col] = Evᵢₜvⱼₜ₊ₛ(i, j, s, N)
        verbose ? println([row, col], " ", Evᵢₜvⱼₜ₊ₛ(i, j, s, N)) : nothing
    end

    # Fill in lower triangle by symmetry
    V = Symmetric(V)
    if verbose
        half = Integer(floor(N * T / 2))
        println("\nLEFT HALF OF COVARIANCE MATRIX")
        println(latexify(V[:, 1:half]))
        println("\nRIGHT HALF OF COVARIANCE MATRIX")
        println(latexify(V[:, (half+1):N*T]))
    end
    return (V)
end


# Negative Log Likelihood (want to minimize)
function nLL(ρ, σₐ, σᵤ, v, N, T)
    V = Σ(ρ,σₐ,σᵤ,N,T)
    nLL = (1/2) * ( v'*V^-1*v + log(det(V)) )
    return(nLL)
end




"""
    mymle(ρstart, σₐstart, σᵤstart, v)

Return likelihood-maximizing values of ρ, σₐ, and σᵤ.
Given vector of residuals, minimize the 
@param ρstart: float, starting value of ρ
@param σₐstart: float, starting value of σₐ
@param σᵤstart: float, starting value of σᵤ
@param v: Nx1 vector of residuals
@param N: int>1, number of units/regions
@param T: int>1, number of time periods
@param lower: float 3-vector, lower bounds for parameters ρ, σₐ, σᵤ
@param upper: float 3-vector, upper bounds for parameters ρ, σₐ, σᵤ
"""
function mymle(ρstart, σₐstart, σᵤstart, v, N, T;
    lower = [1e-2, 1e-2, 1e-2], upper = [1 - 1e-4, Inf, Inf])

    # Starting paramerter values
    params0 = [ρstart, σₐstart, σᵤstart]

    # Function of only parameters (given residuals v)
    fun(ρσₐσᵤ) = nLL(ρσₐσᵤ[1], ρσₐσᵤ[2], ρσₐσᵤ[3], v, N, T)

    # Minimize over parameters
    optimum = optimize(fun, lower, upper, params0)
    # optimum = optimize(nLL, params0, LBFGS(); autodiff = :forward)

    # Return the values
    LL = -optimum.minimum
    ρ, σₐ, σᵤ = optimum.minimizer
    return (ρ, σₐ, σᵤ, LL)
end






"""
    dgp(ρ, σₐ, σᵤ, β, N, T; v₀, μ₀, σ₀)

Return simulated data from the data generating process given paramters.
@param ρ: float in [0,1], decay rate of AR1 emissions process
@param σₐ: float >0, SD of year-specific emissions shock 
@param σᵤ: float >0, SD of region-year-specific emissions shock
@param β: 2-vector, linear and quadratic time trend parameters
@param v₀: float, initial emissions shock of AR1 process (v_t where t=0)
@param b₀: Nx1 array of floats, b₀ᵢ for different regions i in {1, ..., N}
    if not given, then μ₀ and σ₀ are used to pull b₀ᵢ from a random distribution
@param μ₀: float, mean of region-specific fixed effect distribution (b₀ᵢ)
@param σ₀: float >0, SD of region-specific fixed effect distribution (b₀ᵢ)
"""
function dgp(ρ, σₐ, σᵤ, β, N, T;
    v₀ = 0.0, b₀ = nothing, μ₀ = 0, σ₀ = 10, random_seed::Integer = 1234)
    # Initial conditions
    Random.seed!(random_seed)
    b = 1  # unidentified scale parameter

    # Get region-specific fixed effects if not given
    if b₀ === nothing
        b₀ = rand(Distributions.Normal(μ₀, σ₀), N)
    end

    # Random shocks
    αₜ = rand(Distributions.Normal(0, σₐ), T)
    μᵢₜ = rand(Distributions.Normal(0, σᵤ), N * T)
    # assume μᵢₜ is stacked region first:
    # i.e. (i,t=1,1; i,t=2,1; ...; i,t=N-1,T; i,t=N,T)

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

    data = DataFrame(t = repeat(1:T, inner = N),
        i = string.(repeat(1:N, outer = T)),
        b₀ᵢ = repeat(b₀, outer = T),
        αₜ = repeat(αₜ, inner = N),
        μᵢₜ = μᵢₜ,
        vᵢₜ = vᵢₜ,
        vₜ = repeat(vₜ[2:(T+1)], inner = N))

    # Generate the resulting emissions
    data.eᵢₜ = (1 / b) * (data.b₀ᵢ + β[1] * data.t + β[1] * data.t .^ 2 + data.vᵢₜ)

    return (data)
end;





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

    # Go with βse for now
    return(Dict(:β => β,
        :βse => βse,
        :yhat => yhat,
        :resid => resid,
        :βvar => βvar,
        :HC0 => HC0_root,
        :sandwich => s_root
    ))
end


create_dataframe() = DataFrame(
    seed = [], type = [], N = [], T = [],
    b₀ᵢ = [], β = [], ρ = [], σₐ = [], σᵤ = [], iterations = [],
    LL = [], V_norm_finalstep = [], V_norm_true=[], V = []
)

function add_row(df, seed, type, N, T;
    b₀ᵢ=missing, β=missing, ρ=missing, σₐ=missing, σᵤ=missing, iterations=missing,
    LL=missing, V_norm_finalstep=missing, V_norm_true=missing, V=missing)
    push!(df, Dict(
    :seed=>seed, :type=>type, :N=>N, :T=>T,
    :b₀ᵢ=>b₀ᵢ, :β=>β, :ρ=>ρ, :σₐ=>σₐ, :σᵤ=>σᵤ, :LL=>LL, :iterations=>iterations,
    :V_norm_finalstep=>V_norm_finalstep, :V_norm_true=>V_norm_true, :V=>V
    ))
end

function write_df(df; N=2)
    # Save dataframe
    filepath = "../../data/temp/simulation_results_log(N=$N).csv"
    if isfile(filepath)
        CSV.write(filepath, df,append=true)
    else
        CSV.write(filepath, df)
    end
end

function get_covar_names(data)
    ilevels = unique(data.i); len = length(ilevels);
    inames = repeat(["i"], len-1) .* string.(ilevels[2:len])
    varnames = vcat(["intercept"], inames, ["t", "t^2"])
    return(varnames)
end

function get_sample_variances(data)
    σₐ = sum((data.αₜ .- mean(data.αₜ)) .^ 2) / 2 / (T - 1)
    σᵤ = sum((data.μᵢₜ .- mean(data.μᵢₜ)) .^ 2) / (N * T - 1)
    return(σₐ, σᵤ)
end



#######################################################################
#           Iterate to convergence!
#######################################################################
"""
Examine convergence behavior of covariance matrix
"""
function estimate_sigmas(N, T, starting_params;
    n_seeds = 10, iteration_max = 10, convergence_threshold = 1e-16)
    println()
    display([N,T,starting_params])
    @info "Starting estimation with N,T:" N T starting_params n_seeds iteration_max convergence_threshold
    df = create_dataframe()
    # Generate different data to see how convergence behaves
    println("\n", "_"^n_seeds)
    for seed in 1:n_seeds
        print("*")
        @info "seed: " seed
        Random.seed!(seed)
        # Starting parameter values
        ρ = starting_params[:ρ];    β = starting_params[:β]
        σₐ = starting_params[:σₐ];  σᵤ = starting_params[:σᵤ]
        LL = -Inf
        # Simuated data
        data = dgp(ρ, σₐ, σᵤ, β, N, T; random_seed = seed)
        # True values from DGP
        b₀ᵢ = unique(data.b₀ᵢ)
        V_true = Σ(ρ, σₐ, σᵤ, N, T)
        add_row(df, seed, "True", N, T; b₀ᵢ=b₀ᵢ[:], β=β, ρ=ρ, σₐ=σₐ, σᵤ=σᵤ, LL=LL)
        # Observed sample variance from DGP
        σ_observed = get_sample_variances(data)
        add_row(df, seed, "Sample", N, T; σₐ=σ_observed[1], σᵤ=σ_observed[2])
        # Initialize the variance matrix (identity matrix = OLS in first iteration)
        V = Diagonal(ones(length(data.eᵢₜ)))
        # Linear GLS formula
        linear_formula = @formula(eᵢₜ ~ 1 + i + t + t^2)
        varnames = get_covar_names(data)

        Vnorm = Inf; i = 1;
        while Vnorm > convergence_threshold && i <= iteration_max
            # GLS to get residuals v
            global gls = mygls(linear_formula, data, V, N)
            v = vec(gls[:resid])
            # Uncomment the below to remove detrending/demeaning step
            # v = vec(data.vᵢₜ)

            # ML to get parameter estimates ρ, σₐ, σᵤ
            ρ, σₐ, σᵤ, LL = mymle(ρ, σₐ, σᵤ, v, N, T)
            Vold = V
            V = Σ(ρ, σₐ, σᵤ, N, T)
            Vnorm = norm(Vold - V)
            i += 1
        end
        
        β = gls[:β][3:4]; b₀ᵢ[1] = gls[:β][1]; b₀ᵢ[2] = sum(gls[:β][1:2]);
        Vnorm_true = norm(V_true - V)
        add_row(df, seed, "Estimated", N, T;
            b₀ᵢ=b₀ᵢ, β=β, ρ=ρ, σₐ=σₐ, σᵤ=σᵤ, LL=LL, iterations=i-1,
            V_norm_finalstep=Vnorm, V_norm_true=Vnorm_true
        )
    end

    # Save dataframe of simulation results
    write_df(df; N=N)
end


starting_params = Dict(:ρ => 0.85,
    :σₐ => 1,
    :σᵤ => 1,
    :β => [1, 0.1]
)

# println("\nTesting T")
# for T ∈ [5,10,25,50,100]
#     estimate_sigmas(2, T, starting_params; n_seeds=10)
# end

println("\nTesting N")
for N ∈ [3,4]
    estimate_sigmas(N, 50, starting_params; n_seeds=10)
end

println("\nTesting σₐ")
for σₐ ∈ [0.1, 1, 5, 10]
    starting_params = Dict(:ρ => 0.85,
        :σₐ => σₐ,
        :σᵤ => 1,
        :β => [1, 0.1]
    )
    estimate_sigmas(2, 50, starting_params; n_seeds=10)
end

println("\nTesting σᵤ")
for σᵤ ∈ [0.1, 1, 5, 10]
    starting_params = Dict(:ρ => 0.85,
        :σₐ => 1,
        :σᵤ => σᵤ,
        :β => [1, 0.1]
    )
    estimate_sigmas(2, 50, starting_params; n_seeds=10)
end


# Close logging file
close(io)

"""
Notes:
- Try simple case with rho = 1 or zero or other cases Larry mentions

Using the residuals from the true beta and FE, we get a little bit closer 
to small Vnorm between the trueV norm and the estV norm (24.2 instead of 27.4)


# Testing glm function (can't take non-diagonal matrix)
gls = glm(linear_formula, data, Normal(), IdentityLink(), wts=ones(length(data.eᵢₜ)))

# Creating a more analytical gradient
objective(ρ, σₐ, σᵤ, v) = norm(maximand(ρ, σₐ, σᵤ, v));
ρ, σₐ, σᵤ = ρstart, σₐstart, σᵤstart
for k in 1:iteration_max
    grad = gradient(objective(ρ, σₐ, σᵤ, v))
# Could also reverse-mode auto-differentiate to get complete analytical gradient


# Testing symbolic gradient
@variables ρ σₐ σᵤ
V = Σ(2, 3; verbose=false)
Vinv = inv(V)
Vinv = substitute(Vinv, Dict(true => 1))
println(Vinv, "\n")
v = ones(N*T)
nLL(v, N, T) = (1/2)*( v' * inv(Σ(N, T)) * v + log(det(Σ(N, T))) );
nLL = nLL(v, 2, 3)
println(nLL, "\n")
nLLgrad = Symbolics.gradient(nLL, [ρ,σₐ,σᵤ])
println(nLLgrad)











Gls: could decompose using Cholskey and augment the data than feed it into glm 
linear model and use CovarianceMatrices from https://gragusa.org/code/ to get HAC SEs
M.diagonalize()

Iterate many times creating new data and save the vnorm from trueV, see the 
dist of vnorm and distance from true parameter values. Is it just the 
randomness of small sampling?

Also create general V function building function from N, T that returns a 
    function to generate V given params (general function should be used 
        once only for given def of N,T)

Want to compare N=2, small T to N=4, large T. Large T should have better 
convergence? Since we can get more info about variance from more obs



REFERENCES:
If you use Symbolics.jl, please cite this paper
@article{gowda2021high,
  title={High-performance symbolic-numerics via multiple dispatch},
  author={Gowda, Shashi and Ma, Yingbo and Cheli, Alessandro and Gwozdz, Maja and Shah, Viral B and Edelman, Alan and Rackauckas, Christopher},
  journal={arXiv preprint arXiv:2105.03949},
  year={2021}
}


"""

