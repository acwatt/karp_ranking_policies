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

function model2_few_params()

    # Get the simulated data
    # simulate data based on approximate parameters recovered from data
    b₀ = [3.146, -0.454, 3.78, 3.479]
    β = [0.265, 0]
    data = Model.dgp(θ.ρ, θ.σₐ², θ.σᵤ², β, N, T; b₀ = b₀)
    eit = data.eᵢₜ
    et = combine(groupby(data, :t), :eᵢₜ => mean => :eₜ).eₜ

    # Estimate the MLE
    opt4 = optimize(karp_model2(eit, et), MLE())
    opt4.lp
    opt4.values
    opt4.optim_result
    # Then, get the standard errors
    coefdf = coeftable(opt4)
    true_params

    # Try again and fix ρ
    # Initialize dataframe to store the results
    df = DataFrame(UID=Int64[], LL=Float64[], iters=Float64[], 
                    σα²=Float64[], σμ²=Float64[], ρ=Float64[],
                    b01=Float64[], b02=Float64[], b03=Float64[], b04=Float64[],
                    β1=Float64[], β2=Float64[],
                iconverge=Bool[], gconverge=Bool[], fconverge=Bool[],
                gtol=Float64[], ftol=Float64[],
                N=Int64[], T=Int64[], ρfixed=Bool[]
    ); optim_dict = Dict(); UID = 1
    ftol = 1e-40

    # Estimate the MLE
    @showprogress for _ in 1:10
        opt5 = optimize(
            karp_model2(eit, et; ρ=θ.ρ), MLE(), 
            ConjugateGradient(),
            Optim.Options(iterations=50_000, g_tol = 1e-12, f_tol=
            store_trace = true, show_trace=false)
        );
        @show push!(df, (UID, opt5.lp,  # Log likelihood value
            opt5.optim_result.iterations,  # number of iterations
            opt5.values[:σα²], opt5.values[:σμ²], ρ,
            opt5.values[Symbol("b₀[1]")], opt5.values[Symbol("b₀[2]")], opt5.values[Symbol("b₀[3]")], opt5.values[Symbol("b₀[4]")],
            opt5.values[:β₁], opt5.values[:β₂],
            opt5.optim_result.iteration_converged,  # did it hit max iterations?
            opt5.optim_result.g_converged, # did it converge from the gradient?
            opt5.optim_result.f_converged,  # did it converge from the LL value?
            opt5.optim_result.g_abstol,  # gradient tolerance setting
            length(eit) ÷ length(et), length(et), true,  # N, T, ρ is fixed
        ))
        optim_dict[UID] = opt5
        UID += 1
    end

    coeftable(optim_dict[6])
    true_params

end


###############################################  ESIMATE σα², σμ², FIX ρ
# Estimate the MLE on real data, fix ρ
θ = (ρ=0.929347, σₐ²=3.6288344, σᵤ²=3.6288344)  # True paramters
result_fixedρ = estimate_MLE(karp_model3; ρfixed=true, θ=θ, datatype="real", maxiter=50_000, Nseeds=20)
result_fixedρ.best_run
coef_df(result_fixedρ)



######################################################################################
#    Test MLE on simulated data over range of parameters and simulated datasets
######################################################################################
full_sampler = TuringModels.karp_model5(missing, missing)
_res = full_sampler()
_Y = _res.Y
_θ = _res.θ
_m = TuringModels.karp_model5(_Y, missing)
_optim = optimize(_m, MLE(), ConjugateGradient(),
                Optim.Options(iterations=1000, store_trace=true, extended_trace=true)
)
_optim.optim_result
_optim.values
_θ


df = @chain vcat(dfs...) begin
    @subset($"Is Parameter" .== 1)
    @aside param_truths = create_param_truth_vector(_θ, _.Parameter)
    @select(:data_seed, :Parameter, :Truth = param_truths)
    rightjoin(df, on=[:Parameter, :data_seed])
end

function create_param_truth_vector(θ, param_names)
    # if the first element of param_names is a symbol, convert to string
    if isa(param_names[1], Symbol)
        param_names = string.(param_names)
    end
    param_vector = []
    for name in param_names
        # if brackets in name, then it's an element of a vector
        if occursin("[", name)
            # get the vector name
            vector_name = split(name, "[")[1]
            vector_name = strip(vector_name, ['θ', '.'])
            # get the vector
            vector = getfield(θ, Symbol(vector_name))
            # get the index
            index = parse(Int, split(name, "[")[2][1])
            # get the value
            value = vector[index]
            push!(param_vector, value)
        else
            push!(param_vector, getfield(θ, Symbol(strip(name, ['θ', '.']))))
        end
    end
    return param_vector
end

#####################################################################
#  helper functions no longer needed
#####################################################################
# from karp_model4
function initialize_model(model, ρfixed, θ, datatype; kwargs...)
    # Get real or simulated data
    Y = datatype=="real" ? get_MLE_data() : get_MLE_data(θ)
    # Initialize the model
    ρ = ρfixed ? θ.ρ : missing
    return model(Y.eit, Y.et; ρ=ρ, kwargs...)
end


