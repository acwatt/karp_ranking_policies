using Optim

# Define the objective function
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

# Define the initial point
x0 = [0.0, 0.0]

# Use MomentumGradientDescent optimizer
result = optimize(rosenbrock, x0, MomentumGradientDescent())

# Extract the solution and minimum value
solution = result.minimizer
minimum_value = result.minimum

# Print the results
println("Solution: ", solution)
println("Minimum value: ", minimum_value)




function f(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

function h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
end
initial_x = zeros(2)
Optim.minimizer(optimize(f, g!, h!, initial_x, BFGS()))
lower_ = [-1.0, -1.0]; upper_ = [2.0, 2.0];
optimize(f, g!, lower_, upper_, initial_x, Fminbox(GradientDescent()), Optim.Options(iterations=2000))
optimize(f, g!, lower_, upper_, initial_x, Fminbox(MomentumGradientDescent()), Optim.Options(iterations=2000))
optimize(f, g!, lower_, upper_, initial_x, Fminbox(AcceleratedGradientDescent()), Optim.Options(iterations=2000))
Optim.minimizer(optimize(f, g!, h!, lower_, upper_, initial_x, Fminbox(MomentumGradientDescent()), Optim.Options(iterations=2000)))


# Extract the solution and minimum value
solution = result.minimizer
minimum_value = result.minimum

# Print the results
println("Solution: ", solution)
println("Minimum value: ", minimum_value)







############################################################################################################
#    Turing MLE code
############################################################################################################

#######################################
#    Turing attempt for Karp model, no trends
#######################################
#! try estimating a model without trends on simulated data
function model1_no_trends()
    N = 4; T = 60  # 1945-2005
    ρ = 0.8785  # from r-scripts/reproducting_andy.R  -> line 70 result2
    σ²base = 3.6288344  # σᵤ² from test_starting_real_estimation() after rescaling the data to units of 10,000 tons
    θ = (ρ=ρ, σₐ²=σ²base, σᵤ²=σ²base)  # True paramters
    seed = 1234
    data = Model.dgp(θ, N, T, seed)

    @model function karp_model_notrend(e,)

    end
end

#######################################
#  Turing tutorials notebooks
#######################################
# Install TuringTutorials
using Pkg
# pkg"add https://github.com/TuringLang/TuringTutorials"
# pkg"add IJulia"

# Generate notebooks in subdirectory "notebook"
using TuringTutorials
TuringTutorials.weave(; build=(:notebook,))

# Start Jupyter in "notebook" subdirectory
using IJulia
IJulia.notebook(; dir="notebook")








#####################################################################
#    Turing attempt for Karp MLE model, with α and μ error terms
#####################################################################
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



###############################################  ESIMATE σα², σμ², FIX ρ
# Estimate the MLE on real data, fix ρ
θ = (ρ=0.929347, σₐ²=3.6288344, σᵤ²=3.6288344)  # True paramters
result_fixedρ = estimate_MLE(karp_model3; ρfixed=true, θ=θ, datatype="real", maxiter=50_000, Nseeds=20)
result_fixedρ.best_run
coef_df(result_fixedρ)



