""" Script notes
    Script: serial_correlation_tests.jl
    Author: Aaron C Watt
    Date Created: 
    Date Updated:
    Purpose:
    - test/plot autocorrelation coefficient in simulated and real emissions data.


    References:
    [1] http://web.vu.lt/mif/a.buteikis/wp-content/uploads/2019/11/MultivariableRegression_4.pdf pg 77
    [2] https://www.stata.com/manuals13/tsprais.pdf pg 9
    [3] James Mackinnon and Charles Beach, “A Maximum Likelihood Procedure for Regression With Autocorrelated Errors,” Econometrica 46 (January 1, 1978): 51–58, https://doi.org/10.2307/1913644.
    [4] Larry Karp Cap & Trade paper draft, 10/2022 version
"""

using StatsPlots
using DataFramesMeta
using CategoricalArrays
using CSV
import ShiftedArrays as SA
import Statistics as Stat
import Distributions as Dist
using GLM
import Dates

import Revise
Revise.includet("reg_utils.jl")  # mygls_aggregate()


################# GLOBALS ####################
country2num = Dict("USA"=>1, "EU"=>2, "BRIC"=>3, "Other"=>4)



################# HELPER FUNCTIONS ####################
function read_data_real_global()
    T = 60
    filepath = "../../data/sunny/clean/ts_allYears_nation.1751_2014.csv"
    df = @chain CSV.read(filepath, DataFrame) begin
        # Drop first row? (t_lag = 0)
        @rsubset :Year >= 1945 && :Year < 1945+T
        @transform(:t = :time .- 44, 
            :t_lag = parse.(Int, :time_lag) .- 44,
            :ebar = :ebar,
            :ebar_lag = parse.(Float64, :ebar_lag))
        @orderby(:t)
        @select(:t, :ebar, :t_lag, :ebar_lag)
        dropmissing  # Drop missing lagged values
    end
    return(df)
end
function read_data_real_regional(;N=4, T=60)
    filepath = "../../data/sunny/clean/grouped_allYears_nation.1751_2014.csv"
    df = @chain CSV.read(filepath, DataFrame) begin
        @rsubset :Year >= 1945 && :Year < 1945+T && :group ∈ ["USA","EU","BRIC","Other"][1:N]
        @transform(:group = get.(Ref(country2num), :group, missing))
        @select(:t = :time .- 44,
                :i = categorical(:group),
                :eᵢₜ = :CO2)
        @orderby(:t, :i)
    end

    return(df)
end
function read_data_simulated_agg()
    filepath = "../../data/temp/simulated_data N4 T60 rho0.8785 sig_a3.6288344 sig_u3.6288344.csv"
    df = CSV.read(filepath, DataFrame);
    df = @chain CSV.read(filepath, DataFrame) begin
        @by(:t, :ebar = Stat.mean(:eᵢₜ))
        @transform(:ebar_lag = SA.lag(:ebar), :t_lag = SA.lag(:t))
        dropmissing  # Drop missing lagged values
    end
    return(df)
end
function generate_AR1_data()
    T = 60; Tmin = -100; Ntot = T - Tmin + 1
    β = [945.1923, 93.485, 0.0385]; σ²θ = 5; ρ = 0.8785
    v0 = 0; θ = rand(Dist.Normal(0, sqrt(σ²θ)), Ntot)
    v = [ρ*v0 + θ[1]]
    for i ∈ 2:Ntot
        push!(v, ρ*v[i-1] + θ[i])
    end
    df = @chain DataFrame(t = Tmin:T) begin
        @transform(:v = v)
        @transform(:ebar =  β[1] .+ β[2]*:t .+ β[3]*:t.^2 .+ v)
        @transform(:ebar_lag = SA.lag(:ebar))
        @transform(:t_lag = SA.lag(:t))
        @rsubset(:t > 0)
    end
    filepath = "../../data/temp/simulated_data_global T60 rho$ρ sigma$(σ²θ).csv"
    CSV.write(filepath, df)
    return(df)
end

±(x, y) = [x-y, x+y]


################# ANALYSIS FUNCTIONS ####################
"""Return residuals from OLS regression on real data."""
function regression_residuals_real()
    # Read data
    df = read_data_real_global()
    # Run regression
    linear_formula = @formula(ebar ~ 1 + t + t^2)
    ols = lm(linear_formula, df)
    # Return residuals
    return residuals(ols)
end
"""Return residuals from OLS regression on simulated data."""
function regression_residuals_simulated()
    # Read data
    df = read_data_simulated_agg()
    # Run regression
    linear_formula = @formula(ebar ~ 1 + t + t^2)
    ols = lm(linear_formula, df)
    # Return residuals
    return residuals(ols)
end

"""Calculate autocorrelations ρ from residuals."""
function autocorrelations(v)
    # k-th autocorrelation coefficient from reference [1]
    # numerator = Σ_{i=k+1}^N vᵢvᵢ₋ₖ
    numerator(k) = (v[(k+1):end]' * v[1:(end-k)])
    corr_k(k) = numerator(k) / sum(v.^2)
    # Calculate first 20 autocorrelation coefficients
    df = @chain DataFrame(k = 1:20) begin
        @transform :ρ = corr_k.(:k)
    end
    return df
end

"""Return the Durbin-Watson statistic for a vector of residuals. Pg 81 of ref. [1]"""
function DW_stat(v)
    numerator = sum((v[2:end] .- v[1:(end-1)]) .^2)
    denominator = sum(v .^2)
    return numerator / denominator
end


"""Read in real global data and estimate/plot serial correlation coefficients.

    Simulated data created in covariance_matrix.jl
    Code modified from reference [1] in script notes.
"""
function estimate_ρ_real()
    # Get 5% critical value of normal distribution
    z_c = Dist.cquantile(Dist.Normal(), 0.05/2)
    # Run linear regression, get residuals
    v = regression_residuals_real()
    # Calculate autocorrelation coefficients
    df = autocorrelations(v)
    # lower and upper confidence bounds for plotting
    N = length(v)
    lb = -z_c / sqrt(N)
    ub =  z_c / sqrt(N)
    # Plot the correlation coefficients
    @df df plot(:ρ, line=:stem, marker=:dot, markersize=8,
        label="ρₖ", xlabel="k", ylabel="ρ",
        title="Correlation Coefficients for k periods apart")
    hline!([lb], color=:red, label="95% confidence bounds")
    fig = hline!([ub], color=:red, label="")
    # Save plot
    filepath = "../../output/estimation_notes/plotting correlation coefficients/correlation_coef_20k_realdata.png"
    savefig(fig, filepath)
end
"""Read in real global data and estimate/plot serial correlation coefficients.

    Simulated data created in covariance_matrix.jl
    Code modified from reference [1] in script notes.
"""
function estimate_ρ_real()
    # Get 5% critical value of normal distribution
    z_c = Dist.cquantile(Dist.Normal(), 0.05/2)
    # Run linear regression, get residuals
    v = regression_residuals_simulated()
    # Calculate autocorrelation coefficients
    df = autocorrelations(v)
    # lower and upper confidence bounds for plotting
    N = length(v)
    lb = -z_c / sqrt(N)
    ub =  z_c / sqrt(N)
    # Plot the correlation coefficients
    @df df plot(:ρ, line=:stem, marker=:dot, markersize=8,
        label="ρₖ", xlabel="k", ylabel="ρ",
        title="Correlation Coefficients for k periods apart")
    hline!([lb], color=:red, label="95% confidence bounds")
    fig = hline!([ub], color=:red, label="")
    # Save plot
    filepath = "../../output/estimation_notes/plotting correlation coefficients/correlation_coef_20k_simulated.png"
    savefig(fig, filepath)
end

"""Test the Durbin-Watson serial correlation statistic for one lag on real data"""
function DW_test_real() 
    # Run linear regression, get residuals
    v = regression_residuals_real()
    # Caclulate DW statistics
    DW_stat(v)  # 0.21804739204830864
end

"""Test the Durbin-Watson serial correlation statistic for one lag on simulated data"""
function DW_test_simulated()
    # Run linear regression, get residuals
    v = regression_residuals_simulated()
    # Caclulate DW statistics
    DW_stat(v)  # 0.5973463483367228
end



#==============================================================
===============================================================
        Feasible GLS -- Cochrane–Orcutt (CORC) estimator
===============================================================
  See pg 88 of ref [1]
==============================================================#
# TODO: use ρ estimate and σ(?) to est variance matrix and get FGLS resids
# - is this what Larry is talking about in 
function get_FGLS_residuals(df, ρ)
    # Construct covariance matrix
    # Estimate FGLS
    # Get residuals

end



function estimate_ρ_and_σ_CO(data_type)
    println("\nBeginning Cochrane–Orcutt ρ,β,σ estimation on $data_type data.")

    # Read data
    if data_type == "real"
        df = read_data_real_global()
    elseif data_type == "simulated"
        df = read_data_simulated_agg()
    elseif data_type == "generated"
        df = generate_AR1_data()
    end
    # Rename vars for ease of use: ebar -> y, ebar_lag -> ylag, etc
    df = @chain df begin
        @select(:y = :ebar, :ylag = :ebar_lag, :t)
    end
    # initalize ρ for convergence check
    ρold = 0

    # 1. Estimate initial residuals from OLS
    formula1 = @formula(y ~ 1 + t + t^2)
    ols1 = lm(formula1, df)
    β = coef(ols1)
    # Add residuals and lagged residuals to df
    df = @chain df begin
        @transform(:v = residuals(ols1))
        @transform(:vlag = lag(:v))
    end

    # FOR LOOP
    convergence_threshold = 1e-4
    imax = 100
    for i ∈ 1:imax
        # 2. Estimate ρ from OLS regression of residuals on lagged residuals
        # vₜ = ρ vₜ₋₁ + uₜ
        ols2 = lm(@formula(v ~ 0 + vlag), df)
        ρnew = coef(ols2)[1]
        println("i = $i, \tρ = $ρnew, \tβ = $β")

        # 3. Calculate transformed variables
        df = @chain df begin
            @transform(:ytrans = :y .- ρnew * lag(:y),
                    :cons   = 1 - ρnew,
                    :ttrans = :t .- ρnew * lag(:t),
                    :ttrans2 = :t.^2 .- ρnew * lag(:t.^2))
        end

        # 4. Estimate β2 from OLS on transformed variables
        formula4 = @formula(ytrans ~ 0 + cons + ttrans + ttrans2)
        ols3 = lm(formula4, df)
        β = coef(ols3)

        # Check for convergence of ρ
        if abs(ρold - ρnew) < convergence_threshold
            println("Convergence Reached!")
            ρold = ρnew
            break
        end
        ρold = ρnew

        # 5. Estimate new residuals using β2 on non-transformed variables
        X = StatsModels.modelmatrix(formula1.rhs, df)
        y = StatsModels.modelmatrix(formula1.lhs, df)
        v = (y .- X*coef(ols3))[:,1]
        df = @chain df begin
            @transform(:v = v)
            @transform(:vlag = lag(:v))
        end

        # Repeat until convergence of ρ or maximum iterations reached
        if i == imax
            println("Reached maximum iterations: No Convergence.")
        end
    end

    fgls_results = mygls_aggregate(formula1, df, ρold)
    # Calculate σ^2 = (1-ρ^2)*Var(v)
    σ² = (1-ρold^2)*Stat.var(df.v)
    results = Dict(:ρ => ρold, :σ² => σ², :β => β,
                :σ²v => fgls_results[:σ²v], :σ²θ => fgls_results[:σ²θ])

    println(["$k = $(round.(v,digits=4))" for (k,v) in results])
    return results
end

real_params_CO = estimate_ρ_and_σ_CO("real")
# ["ρ = 0.9056", "σ²μ = 8293.1128", "σ²ε = 46080.9161", "σ² = 13742.7899", "β = [300.6616, 125.7893, -0.3339]"]

sim_params_CO = estimate_ρ_and_σ_CO("simulated")
# ["ρ = 0.6365", "σ²μ = 3.7377", "σ²ε = 6.2833", "σ² = 5.0445", "β = [-3.4966, 0.5767, -0.0059]"]

sim_params_CO = estimate_ρ_and_σ_CO("generated");
#     ["ρ = 0.7563", "σ²μ = 5.121", "σ²ε = 11.9638", "σ² = 5.1806", "β = [11.4331, 1.707, 0.0072]"]
# True: ρ = 0.8785                                    σ² = 5         β = [10,     2,       0.002];



#==============================================================
===============================================================
        Feasible GLS -- Prais-Winsten estimator
===============================================================
  See pg 9 of ref [2]
==============================================================#
function estimate_ρ_and_σ_PW(data_type; verbose=false)
    vprint(s) = verbose ? println(s) : nothing

    vprint("\nBeginning Prais-Winsten ρ,β,σ estimation on $data_type data.")

    # Read data
    if data_type == "real"
        df = read_data_real_global()
    elseif data_type == "simulated"
        df = read_data_simulated_agg()
    elseif data_type == "generated"
        df = generate_AR1_data()
    end
    # Rename vars for ease of use: ebar -> y, ebar_lag -> ylag, etc
    df = @chain df begin
        @select(:y = :ebar, :ylag = :ebar_lag, :t)
    end
    # initalize ρ for convergence check
    ρold = 0

    # 1. Estimate initial residuals from OLS
    formula1 = @formula(y ~ 1 + t + t^2)
    ols1 = lm(formula1, df)
    β = coef(ols1)
    # Add residuals and lagged residuals to df
    df = @chain df begin
        @transform(:v = residuals(ols1))
        @transform(:vlag = lag(:v))
    end

    # FOR LOOP
    convergence_threshold = 1e-4
    imax = 100
    for i ∈ 1:imax
        # 2. Estimate ρ from OLS regression of residuals on lagged residuals
        # vₜ = ρ vₜ₋₁ + uₜ
        ols2 = lm(@formula(v ~ 0 + vlag), df)
        ρ = coef(ols2)[1]
        vprint("i = $i, \tρ = $ρ, \tβ = $β")

        # 3. Calculate Cochrane–Orcutt transformed variables
        df = @chain df begin
            @transform(:ytrans = :y .- ρ * lag(:y),
                    :cons   = 1 - ρ,
                    :ttrans = :t .- ρ * lag(:t),
                    :ttrans2 = :t.^2 .- ρ * lag(:t.^2))
        end
        # Add first-row transformation from Prais-Winsten method
        df = @transform df @byrow begin
            :ytrans = :t==1 ? sqrt(1-ρ^2)*:y : :ytrans
            :cons   = :t==1 ? sqrt(1 - ρ^2) : :cons
            :ttrans = :t==1 ? sqrt(1 - ρ^2)*:t : :ttrans
            :ttrans2 = :t==1 ? sqrt(1 - ρ^2)*:t.^2 : :ttrans2
        end

        # 4. Estimate β2 from OLS on transformed variables
        formula4 = @formula(ytrans ~ 0 + cons + ttrans + ttrans2)
        ols3 = lm(formula4, df)
        β = coef(ols3)

        # Check for convergence of ρ
        if abs(ρ - ρold) < convergence_threshold
            vprint("Convergence Reached!")
            break
        end
        ρold = ρ

        # 5. Estimate new residuals using β2 on non-transformed variables
        X = StatsModels.modelmatrix(formula1.rhs, df)
        y = StatsModels.modelmatrix(formula1.lhs, df)
        v = (y .- X*coef(ols3))[:,1]
        df = @chain df begin
            @transform(:v = v)
            @transform(:vlag = lag(:v))
        end

        # Repeat until convergence of ρ or maximum iterations reached
        if i == imax
            vprint("Reached maximum iterations: No Convergence.")
        end
    end

    fgls_results = mygls_aggregate(formula1, df, ρold)
    # Calculate σ^2 = (1-ρ^2)*Var(v)
    σ² = (1-ρold^2)*Stat.var(df.v)
    results = Dict(:ρ => ρold, :σ² => σ², :β => β,
                :σ²v => fgls_results[:σ²v], :σ²θ => fgls_results[:σ²θ])

    vprint(["$k = $(round.(v,digits=4))" for (k,v) in results])

    return results
end

real_params_PW = estimate_ρ_and_σ_PW("real");
println("real_params_PW: $real_params_PW")
# ["ρ = 0.9277", "σ²μ = 8268.1726", "σ²ε = 59298.9209", "σ² = 8332.6223", "β = [960.1514, 91.7068, 0.0689]"]

sim_params_PW = estimate_ρ_and_σ_PW("simulated");
println("sim_params_PW: $sim_params_PW")
# ["ρ = 0.6365", "σ²μ = 3.7377", "σ²ε = 6.2836", "σ² = 5.0442", "β = [-3.4966, 0.5767, -0.0059]"]

gen_params_PW = estimate_ρ_and_σ_PW("generated");
println("gen_params_PW: $gen_params_PW")
#     ["ρ = 0.8305", "σ²μ = 4.8378", "σ²ε = 15.5927", "σ² = 4.9299", "β = [11.5403, 2.1223, -0.0008]"]
# True: ρ = 0.8785                                     σ² = 5         β = [10,      2,      0.002]

ρvec_PW = [estimate_ρ_and_σ_PW("generated")[:ρ] for i in 1:10_000]
histogram(ρvec_PW, label="", title="Dist of ρ estimates (PW estimator)", bins=100)
vline!([0.8785], color="red", label="True ρ")
vline!([Stat.quantile(ρvec_PW, 0.95)], color="green", label="95% quantile")




# data_type = "real"
# # Read data
#     if data_type == "real"
#         df = read_data_real_global()
#         ρ = real_params_PW[:ρ]
#     elseif data_type == "simulated"
#         df = read_data_simulated_agg()
#     end
#     # Rename vars for ease of use: ebar -> y, ebar_lag -> ylag, etc
#     df = @chain df begin
#         @select(:y = :ebar, :ylag = :ebar_lag, :t)
#     end
#     # Define covariance weighting matrix Σ = σ²Ω
#     formula = @formula(y ~ 1 + t + t^2)
#     mygls_aggregate(formula, df, ρ)
#     # :β => [960.061; 91.7179; 0.0687466;;]
#     # :σ²ε      => 59298.9
#     # :σ²μ      => 8268.17
#     fds








#==============================================================
===============================================================
        MLE -- Mackinnon-Beach estimator
===============================================================
  See pg 53 of ref [3]
    "The technique proceeds by alternately maximizing [the LL] with
    respect to β, ρ held fixed, and maximizing [the LL] with respect
    ρ, β held fixed. This usually begins with ρ equal to zero, and 
    ends when two successive values of ρ are sufficiently close. Two
    separate maximizations must be performed at each iteration. The 
    solution to maximizing [the LL] with respect to β, ρ held fixed
    is in the β from PW estimation on the transformed data.
==============================================================#
function myround(v; digits=3) 
    try
        return round(v, digits=digits)
    catch e
        if isa(e, MethodError)
            return myround.(v, digits=digits)
        else
            error(e)
        end
    end
end

function estimate_ρ_and_σ_MB(data_type; verbose=false)
    vprint(s) = verbose ? println(s) : nothing

    vprint("\nBeginning Mackinnon-Beach ρ,β,σ estimation on $data_type data.")

    # Read data
    data_type = "real"
    if data_type == "real"
        df = read_data_real_global()
    elseif data_type == "simulated"
        df = read_data_simulated_agg()
    elseif data_type == "generated"
        df = generate_AR1_data()
    end
    # Rename vars for ease of use: ebar -> y, ebar_lag -> ylag, etc
    df = @chain df begin
        @select(:y = :ebar, :ylag = :ebar_lag, :t)
    end
    formula1 = @formula(y ~ 1 + t + t^2)
    X = StatsModels.modelmatrix(formula1.rhs, df)
    y = StatsModels.modelmatrix(formula1.lhs, df)
    # initalize ρ for convergence check
    ρ = 0
    T = size(df)[1]
    β = [0,0,0]

    # FOR LOOP
    convergence_threshold = 1e-8
    imax = 1000
    for i ∈ 1:imax
        ρold = ρ        

        # 3. Calculate Cochrane–Orcutt transformed variables
        df = @chain df begin
            @transform(:ytrans = :y .- ρ * lag(:y),
                    :cons   = 1 - ρ,
                    :ttrans = :t .- ρ * lag(:t),
                    :ttrans2 = :t.^2 .- ρ * lag(:t.^2))
        end
        # Add first-row transformation from Prais-Winsten method
        df = @transform df @byrow begin
            :ytrans  = :t==1 ? sqrt(1 - ρ^2)*:y    : :ytrans
            :cons    = :t==1 ? sqrt(1 - ρ^2)       : :cons
            :ttrans  = :t==1 ? sqrt(1 - ρ^2)*:t    : :ttrans
            :ttrans2 = :t==1 ? sqrt(1 - ρ^2)*:t.^2 : :ttrans2
        end

        # 4. Estimate β from OLS on transformed variables; eqn (3)
        formula4 = @formula(ytrans ~ 0 + cons + ttrans + ttrans2)
        ols3 = lm(formula4, df)
        β = coef(ols3)

        # 5. Estimate new residuals using β on non-transformed variables
        v = (y .- X*β)[:,1]
        df = @chain df begin
            @transform(:v = v)
            @transform(:vlag = lag(:v))
        end

        # 6. Calculate ρ by finding root of eqn (4)
        A₁ = first(df.v)
        Aₜ = df.v[2:end]
        Aₜ₋₁ = df.vlag[2:end]
        denom = (T-1)*(sum(Aₜ₋₁.^2) - A₁^2)
        a = -(T-2)*sum(Aₜ .* Aₜ₋₁) / denom
        b = ((T-1)A₁^2 - T*sum(Aₜ₋₁.^2) - sum(Aₜ.^2) ) / denom
        c = T*sum(Aₜ .* Aₜ₋₁) / denom
        p = b - a^2/3; q = c - a*b/3 + 2a^3/27;
        ϕ = acos( (q*sqrt(27)) / (2p*sqrt(-p)) )
        ρ = -2*sqrt(-p/3)*cos(ϕ/3 + π/3) - a/3
        # vprint("i = $i, \tρ = $ρ, \tβ = $β")

        # Check for convergence of ρ
        if abs(ρ - ρold) < convergence_threshold
            vprint("Convergence Reached!")
            break
        end

        # Repeat until convergence of ρ or maximum iterations reached
        if i == imax
            vprint("Reached maximum iterations: No Convergence.")
        end
    end

    fgls_results = mygls_aggregate(formula1, df, ρ)
    ρvar = (1-ρ^2)/T
    ρse = sqrt(ρvar)
    degf = T - size(β)[1]  # degrees of freedom
    tcrit = Dist.quantile(Dist.TDist(degf), 0.975)
    ρCI95 = ρ ± tcrit*ρse


    """Build Variance-Covariance of Mackinnon-Beach 1978 parameters from Appendix (see ref [3]).

        based on the AR(1) model: y = Xβ + v; vₜ = ρvₜ₋₁ + θₜ; var(θₜ) = σ²
    """
    function build_param_varcov_matrix(β, σ², ρ, T, ε, X)
        A = T/(2σ²^2)
        B = (1+ρ^2)/(1-ρ^2)^2 + (1/σ²)*sum(ε[2:(end-1)].^2)
        C = (1/σ²)*(ρ / (1-ρ^2))
        k = size(β)[1]
        D = [(1-ρ^2)X[1,i]X[1,j] + sum((X[2:end,i]-ρ*X[1:(end-1),i]).*(X[2:end,j]-ρ*X[1:(end-1),j])) for i ∈ 1:k, j ∈ 1:k] ./ σ²
        wide0 = zeros(1, size(D)[2])
        long0 = zeros(size(D)[1], 1)
        M = [A     C     wide0
             C     B     wide0
             long0 long0 D    ]
        Vcov = inv(M)

        Var = [Vcov[i,i] for i ∈ 1:size(Vcov)[1]]
        vprint("Variance-Covariance matrix for (σ², ρ, β₀, β₁, β₂)")
        display(Vcov)
        σ²se = Var[1] ^0.5
        ρse = Var[2] ^0.5
        βse = Var[3:end] .^0.5

        return Dict(:σ²se => σ²se,
                    :ρse => ρse,
                    :βse => βse
        )
    end
    σ²θ = fgls_results[:σ²θ]
    Vcov_results = build_param_varcov_matrix(β, σ²θ, ρ, T, df.v, X)
    
    println("\nfgls_results:")
    display(fgls_results)
    
    σ²θse = Vcov_results[:σ²se]
    σ²θCI95 = σ²θ ± tcrit*σ²θse

    βse = Vcov_results[:βse]
    βCI95 = β .± tcrit*βse

    results = [
        Dict(:parameter => :ρ,   :point_est => ρ,    :SE => ρse,    :CIlower => ρCI95[1],    :CIupper => ρCI95[2]),
        Dict(:parameter => :B₀,  :point_est => β[1], :SE => βse[1], :CIlower => βCI95[1][1], :CIupper => βCI95[1][2]),
        Dict(:parameter => :β₁,  :point_est => β[2], :SE => βse[2], :CIlower => βCI95[2][1], :CIupper => βCI95[2][2]),
        Dict(:parameter => :β₂,  :point_est => β[3], :SE => βse[3], :CIlower => βCI95[3][1], :CIupper => βCI95[3][2]),
        Dict(:parameter => :σ²θ, :point_est => σ²θ,  :SE => σ²θse, :CIlower  => σ²θCI95[1],  :CIupper =>  σ²θCI95[2]),
        Dict(:parameter => :σ²v, :point_est => fgls_results[:σ²v])]
    println("\nMB_results:")
    display(results)
    return results
end

function build_results_df(li)
    dfs = [DataFrame(d) for d in li]
    df = select!(reduce(vcat, dfs, cols = :union), [:parameter, :point_est, :SE, :CIlower, :CIupper])
    rename!(df, [:Parameter, Symbol("Point Est"), :SE, Symbol("CI Lower"), Symbol("CI Upper")])
    return df
end

function save_results_MB(li, data_type)
    df = build_results_df(li)
    date = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM")
    filepath = "../../output/estimation_notes/estimating rho/rho_estimation_MB_$data_type$date.csv"
    CSV.write(filepath, df)
end

real_params_MB = estimate_ρ_and_σ_MB("real"; verbose=true);
println("real_params_MB: $real_params_MB")
# ["ρ = 0.9127", "σ²μ = 8279.8607", "σ²ε = 49596.924", "σ² = 9637.4533", "β = [945.1923, 93.485, 0.0385]"]

sim_params_MB = estimate_ρ_and_σ_MB("simulated");
println("sim_params_MB: $sim_params_MB")
# ["ρ = 0.9998", "σ²μ = 4.0529", "σ²ε = 9022.2246", "σ² = 1.6457", "β = [-282.4142, 5.7708, -0.0343]"]
# did not converge

gen_params_MB = estimate_ρ_and_σ_MB("generated");
println("gen_params_MB: $gen_params_MB")
#     ["ρ = 0.8635", "σ²μ = 4.7284", "σ²ε = 18.5836", "σ² = 5.1545", "β = [14.145, 1.5167, 0.0106]"]
# True: ρ = 0.8785                                     σ² = 5         β = [10,     2,      0.002]

test_sampling_distribution = false
if test_sampling_distribution
    ρvec_MB = []; ρwidth_MB = []
    iters = 10_000
    println("."^Integer(iters/100))
    for i in 1:iters
        i%100==1 ? print(".") : nothing
        res = estimate_ρ_and_σ_MB("generated")
        push!(ρvec_MB, res[:ρ])
        push!(ρwidth_MB, 2*1.96*res[:ρse])
    end
    pMB = histogram(ρvec_MB, label="", title="Dist of ρ estimates (MB estimator)", bins=50)
    vline!([0.8785], color="red", label="True ρ")
    vline!([Stat.quantile(ρvec_MB, 0.95)], color="green", label="95% quantile")

    pMB1 = histogram(ρwidth_MB, label="", title="Width of ρ estimate 95% CIs (MB estimator)", bins=50)
    # vline!([Stat.quantile(ρvec_MB, 0.95)], color="green", label="95% quantile")

    ρupper = [ρ+0.5*w for (ρ,w) in zip(ρvec_MB, ρwidth_MB)]
    histogram(ρupper, label="", title="Upper bound of ρ estimate 95% CIs (MB estimator)", bins=50)
    pMB2 = vline!([0.8785], color="red", label="True ρ")

    import StatsBase as SB
    cdf_func = SB.ecdf(ρupper)
    plot(cdf_func, label="Empirical CDF of upper bound of 95% CIs")
    vline!([0.8785], color="red", label="True ρ")
    pMB3 = hline!([cdf_func(0.8785)], color="green", label="$(round(cdf_func(0.8785)*100, digits=2))% of upper bounds below true ρ")
end





#==============================================================
===============================================================
        Equation 77 - estimate σᵤ
===============================================================
See pg 23 of the appendix of ref [4], starting after (75)
    - 
==============================================================#
# Load the data
T = 60; n = 4
df = read_data_real_regional(N=n, T=T)
# Calculate time-aggregate emmissions for each region
df_AG = @chain df begin
    @by(:i, :eiAG = sum(:eᵢₜ))
    @aside eAG = sum(_.eiAG)
    @transform(:boiB = (:eiAG .- eAG/n)/T)
end

df_t = @chain df begin
    @by(:t, :etbar = Stat.mean(:eᵢₜ))
    leftjoin(df, _, on = :t)
    @transform(:y = :eᵢₜ .- :etbar)
    @rsubset(:i != 4)
end

Ω = n*I - ones(n-1, n-1)
s = ones(T)
y = df_t.y

# Calculate σ²ᵤ/(nb²)
using Kronecker: ⊗
numerator = y' * ((I - s*s'/T) ⊗ inv(Ω)) * y
denominator = (n-1)T - n
σ²μn = numerator/denominator

# Save results
push!(real_params_MB, 
    Dict(:parameter => :σ²μ, :point_est => σ²μn),
    Dict(:parameter => Symbol("b₀₁-B₀"), :point_est => @rsubset(df_AG, :i==1).boiB[1]),
    Dict(:parameter => Symbol("b₀₂-B₀"), :point_est => @rsubset(df_AG, :i==2).boiB[1]),
    Dict(:parameter => Symbol("b₀₃-B₀"), :point_est => @rsubset(df_AG, :i==3).boiB[1]),
    Dict(:parameter => Symbol("b₀₄-B₀"), :point_est => @rsubset(df_AG, :i==4).boiB[1]))
save_results_MB(real_params_MB, "real-data")






