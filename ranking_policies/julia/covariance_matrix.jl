# Loading packages can often take a long time, but Julia is very fast once packages are loaded
using Zygote  # for reverse-mode auto-differentiation
using Random  # Random numbers
using Distributions
using DataFrames
using LinearAlgebra  # Diagonal
using StatsModels  # formulas and modelmatrix
using GLM  # OLS (lm) and GLS (glm)
using Optim  # optimize (minimizing)


#######################################################################
#           Functions
#######################################################################

# Building the covariance matrix
N=2; T=3
A(ρ, σₐ, σᵤ) = 1/(1 - ρ^2)*σₐ^2 + (1 + ρ^2/(N*(1 - ρ^2)))*σᵤ^2;
B(ρ, σₐ, σᵤ) = 1/(1 - ρ^2)*σₐ^2 + (ρ^2/(N*(1 - ρ^2)))*σᵤ^2;
C(ρ, σₐ, σᵤ) = ρ/(1 - ρ^2)*σₐ^2 + (ρ/N + ρ^3/(N*(1 - ρ^2)))*σᵤ^2;
E(ρ, σₐ, σᵤ) = ρ*C(ρ, σₐ, σᵤ);

function ∑(ρ, σₐ, σᵤ)
  [A(ρ, σₐ, σᵤ) B(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) E(ρ, σₐ, σᵤ) E(ρ, σₐ, σᵤ)
   B(ρ, σₐ, σᵤ) A(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) E(ρ, σₐ, σᵤ) E(ρ, σₐ, σᵤ)
   C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) A(ρ, σₐ, σᵤ) B(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ)
   C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) B(ρ, σₐ, σᵤ) A(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ)
   E(ρ, σₐ, σᵤ) E(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) A(ρ, σₐ, σᵤ) B(ρ, σₐ, σᵤ)
   E(ρ, σₐ, σᵤ) E(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) C(ρ, σₐ, σᵤ) B(ρ, σₐ, σᵤ) A(ρ, σₐ, σᵤ)]
end

# Negative Log Likelihood (want to minimize)
nLL(ρ, σₐ, σᵤ, v) = (1/2)*( v' * inv(∑(ρ, σₐ, σᵤ)) * v + log(det(∑(ρ, σₐ, σᵤ))) );



"""
    mymle(ρstart, σₐstart, σᵤstart, v)

Return likelihood-maximizing values of ρ, σₐ, and σᵤ.
Given vector of residuals, minimize the 
@param ρstart: float, starting value of ρ
@param σₐstart: float, starting value of σₐ
@param σᵤstart: float, starting value of σᵤ
@param v: Nx1 vector of residuals
"""
function mymle(ρstart, σₐstart, σᵤstart, v;
    lower = [1e-2, 1e-2, 1e-2], upper = [1-1e-4, Inf, Inf])

    # Starting paramerter values
    params0 = [ρstart, σₐstart, σᵤstart]

    # Function of only parameters (given residuals v)
    fun(ρσₐσᵤ) = nLL(ρσₐσᵤ[1], ρσₐσᵤ[2], ρσₐσᵤ[3], v)

    # Minimize over parameters
    optimum = optimize(fun, lower, upper, params0)
    # optimum = optimize(nLL, params0, LBFGS(); autodiff = :forward)

    # Return the values
    MLE = -optimum.minimum
    ρ, σₐ, σᵤ = optimum.minimizer
    return(ρ, σₐ, σᵤ, MLE)
end


"""
    dgp(ρ, σₐ, σᵤ, μ₀, σ₀, β, N, T)

Return simulated data from the data generating process given paramters.
@param ρ: float in [0,1], decay rate of AR1 emissions process
@param σₐ: float >0, SD of year-specific emissions shock 
@param σᵤ: float >0, SD of region-year-specific emissions shock
@param μ₀: float, mean of region-specific fixed effect distribution
@param σ₀: float >0, SD of region-specific fixed effect distribution
@param β: 2-vector, linear and quadratic time trend parameters
""";
function dgp(ρ, σₐ, σᵤ, β, N, T)
    # Initial conditions
    Random.seed!(1234)
    b = 1  # unidentified scale parameter
    v₀ = 0.0  # initial emissions shock
    μ₀ = 0  # mean of region-specific fixed effect distribution (b₀ᵢ)
    σ₀ = 10 # SD of region-specific fixed effect distribution (b₀ᵢ)
    
    # Get region-specific fixed effects
    b₀ = rand(Distributions.Normal(μ₀, σ₀), N)
    
    # Random shocks
    αₜ = rand(Distributions.Normal(0, σₐ), T)
    μᵢₜ = rand(Distributions.Normal(0, σᵤ), N*T)
    # assume μᵢₜ is stacked region first:
    # i.e. (i,t=1,1; i,t=2,1; ...; i,t=N-1,T; i,t=N,T)
    
    # Fill in the aggregate shocks
    vₜ = [v₀]; vᵢₜ = [];
    for t in 1:T
        s = 0  # this period's sum of shocks
        for i in 1:N
            # Note that vₜ index starts at 1 instead of 0, so it's ahead of the others
            append!(vᵢₜ, ρ*vₜ[t] + αₜ[t] + μᵢₜ[(t-1)*N+i])
            s += last(vᵢₜ)
        end
        append!(vₜ, s/N)  # Aggregate shock = average of this period's shocks
    end
    
    data = DataFrame(t = repeat(1:T, inner=N),
                     i = string.(repeat(1:N, outer=T)),
                     b₀ᵢ = repeat(b₀, outer=T),
                     αₜ = repeat(αₜ, inner=N),
                     μᵢₜ = μᵢₜ,
                     vᵢₜ = vᵢₜ,
                     vₜ = repeat(vₜ[2:(T+1)], inner=N))
    
    # Generate the resulting emissions
    data.eᵢₜ = (1/b)*(data.b₀ᵢ + β[1]*data.t + β[1]*data.t.^2 + data.vᵢₜ)
    
    return(data)
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
    X = StatsModels.modelmatrix(formula.rhs, df);
    y = StatsModels.modelmatrix(formula.lhs, df)
    XX = inv(X'X)
    XWX = X'*inv(W)*X
    NT, k = size(X)
    β = inv(XWX) * (X'*inv(W)*y)
    yhat = X*β
    resid = y - yhat
    XeX = X'*Diagonal(vec(resid.^2))*X
    # Variance estimators
    # Hansen eq 4.16 (sandwich estimator)
    s = XX * XWX * XX
    s_root = diag(s).^0.5
    # White estimator (rsusmel pg 11)
    HC0 = XX * XeX * XX
    HC0_root = diag(HC0).^0.5
    # Greene (9-13)
    βvar = inv(XWX) / (NT - N - k)
    βse = diag(βvar).^0.5
    
    # TODO: Add Newey-West HAC SE estimator.
    #   pg. 20 of bauer.uh.edu/rsusmel/phd/ec1-11.pdf

    # Go with βse for now
    return(Dict(:β => β,
                :βse => βse,
                :yhat => yhat,
                :resid => resid,
                :βvar => βvar,
                :HC0 => HC0_root,
                :sandwich => s_root))
end



#######################################################################
#           Iterate to convergence!
#######################################################################

Random.seed!(1234)
iteration_max = 10
# Starting parameter values
ρ=0.85; σₐ=1; σᵤ=1; β=[1, 0.1]; LL = -Inf
ρtrue, σₐtrue, σᵤtrue, βtrue = ρ, σₐ, σᵤ, β
# Parameter bounds
lower = [1e-4, 1e-2, 1e-2]
upper = [1-1e-4, Inf, Inf]
# Simuated data
N = 2  # Number of regions
T = 3  # Number of time periods
v₀ = 0.0  # initial emissions shock
μ₀ = 0  # mean of region-specific fixed effect distribution
σ₀ = 10 # SD of region-specific fixed effect distribution
data = dgp(ρ, σₐ, σᵤ, β, N, T);
vtrue = vec(data.vᵢₜ)
# Initialize the variance matrix (identity matrix = OLS in first iteration)
V = Diagonal(ones(length(data.eᵢₜ)));
V_true = ∑(ρ, σₐ, σᵤ)
# Linear GLS formula
linear_formula = @formula(eᵢₜ ~ 1 + i + t + t^2);


for k in 1:iteration_max
    println("\niteration ", k, " LL: ", LL)
    println("Voldvs Vtrue: ", norm(V_true - V))
    println("ρ: ", ρ, " σₐ: ", σₐ, " σᵤ: ", σᵤ)
    # GLS to get residuals v
    gls = mygls(linear_formula, data, V, N)
    println("β: ", gls[:β])
    v = vec(gls[:resid]);
    # v = vtrue
    # ML to get parameter estimates ρ, σₐ, σᵤ
    global ρ, σₐ, σᵤ, LL = mymle(ρ, σₐ, σᵤ, v)
    Vold = V
    global V = ∑(ρ, σₐ, σᵤ)
    Vnorm = norm(Vold - V)
    println("Vold vs Vnew: ", Vnorm)
end




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
"""

