# This file is designed to create highly parallel simuluations and estimation of the model using Turing.
# For distriubted computing on Savio cluster

using Distributed, SlurmClusterManager
addprocs(SlurmManager())  # this will only work when run on Savio

# Remove environment on local machine
@info "Removing environment files: $(string(@__DIR__, "/Manifest.toml"))"
try; rm(string(@__DIR__, "/Manifest.toml")); catch e; end
try; rm(string(@__DIR__, "/Project.toml")); catch e; end
@info "Done removing environment files."

# Resetup the environment files so they are compatible with the remote nodes
using Pkg
Pkg.activate(@__DIR__)
Pkg.add(["SMTPClient", "CSV", "Turing", "Optim", "DynamicHMC", "Bijectors",
         "DataFrames", "Dates", "ProgressMeter", "LoggingExtras", "Random"])
@info "Done creating new environment."

# Instantiate/add environment in all processes
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end
@info "Done with project activation."

# Wait until all packages have been added, then load needed packages
# Include files in global scope first, then everywhere according to https://github.com/JuliaLang/julia/issues/16788#issuecomment-226977022
using Turing
using DataFrames
using Dates
using Optim
using ProgressMeter
using Random
using Logging, LoggingExtras
using CSV
include("../Model/TuringModels.jl")  # TuringModels
include("../Utilities/Communications.jl")  # send_txt
@everywhere begin
    using Turing
    using DataFrames
    using Dates
    using Optim
    using ProgressMeter
    using Random
    using Logging, LoggingExtras
    using CSV
    include("../Model/TuringModels.jl")  # TuringModels
    include("../Utilities/Communications.jl")  # send_txt
end
@info "Done with package loads."
@info "\n"^10
@info "*"^60
@info "Starting TuringSimulationsParallel.jl"

################################ Helper Functions ################################
"""Convert a NamedTuple or NamedVector to a list of key-value pairs."""
itemize(nt) = [(k, v) for (k, v) in zip(keys(nt), values(nt))]

"""Sample parameters from prior given σα², σμ²"""
param_sampler(σα², σμ²) = TuringModels.karp_model5_parameters((; σα², σμ²))
"""Convert θ with b₀ vector to θ with b₀₁, b₀₂, b₀₃, b₀₄"""
function param_flatten(θ)
    θ2 = NamedTuple(p for p in itemize(θ) if p[1] != :b₀)
    θ3 = (; b₀₁ = θ.b₀[1], b₀₂ = θ.b₀[2], b₀₃ = θ.b₀[3], b₀₄ = θ.b₀[4], θ2...)
    return θ3
end
"""Convert θ with b₀₁, b₀₂, b₀₃, b₀₄ to θ with b₀ vector"""
function param_vectorize(θin)
    θ = isa(θin, DataFrame) ? first(θin) : θin
    b₀ = [θ.b₀₁, θ.b₀₂, θ.b₀₃, θ.b₀₄]
    θ2 = NamedTuple(p for p in itemize(θ) if !occursin("b₀", String(p[1])))
    θ3 = (; b₀, θ2...)
    return θ3
end

"""Sample data from model given true parameters"""
data_sampler(θ) = TuringModels.karp_model5(missing, θ)

"""
    `initialize_model(model; θ=missing, seed=1234, kwargs...)`
    `initialize_model(model, Prior::Turing.Prior; kwargs...)`

    Return model initialized with data, or for prior sampling.

    `initialize_model(model; θ=missing, kwargs...)`
    - `model` = Turing model function
    - `θ` = NamedTuple, parameters to generate simulated data with
    - If `θ = missing`, observed data is used, parameters are to be estimated.


    # Examples
    ## Estimate MLE parameters of karp_model5 using observed data
    ```julia
    m = initialize_model(TuringModels.karp_model5)
    opt = optimize(m, MLE(), ConjugateGradient())
    opt.values
    ```

    ## Set "true" parameters to generate data
    ```julia
    θ = (; b₀=[0.1, 0.1, 0.1, 0.1], β₁=0.1, β₂=0.1, σα²=0.1, σμ²=0.1, ρ=0.1, v0=0)
    # Generate simulated data and estimate MLE parameters of karp_model5
    m = initialize_model(TuringModels.karp_model5; θ)
    opt = optimize(m, MLE(), ConjugateGradient())
    opt.values
    ```

    ## Sample data and parameters from prior
    ```julia
    # Generate simulated data and estimate MLE parameters of karp_model5
    m = initialize_model(TuringModels.karp_model5, Prior())
    data = m()
    ```
"""
function initialize_model(model; θ=missing, seed=1234, kwargs...)
    # If θ is missing, Y = observed data, else Y = simulated data from θ
    Y = ismissing(θ) ? get_MLE_data() : get_MLE_data(model, θ; seed)
    return model(Y, missing; kwargs...)
end

function try_catch_optim(m; maxiter=100, maxtime=60*60*24)
    try
        opt = optimize(
                m,  # model
                MLE(),  # parameter estimation method
                ConjugateGradient(),  # optimization algorithm
                Optim.Options(
                    iterations=maxiter,
                    store_trace=true,
                    extended_trace=true,
                    time_limit=maxtime
                )
        )
        return opt
    catch e
        println("Error: ", e)
        println("Terminating optimization")
        @error "Optimization Error caused optim termination"
        @info "$e"
        return missing
    end
end

function get_estimated_params(opt)
    # Get the estimated parameters from the optimization results
    val = opt.values
    name = map(s -> lstrip(s, ['θ','.']), String.(keys(opt.values.dicts[1])))
    # Turn b0 params into an array for adding to the dataframe of results
    b0 = [v for (n,v) in zip(name, val) if occursin("b₀", n)]
    b0names = ["b₀₁", "b₀₂", "b₀₃", "b₀₄"]
    d = Dict(zip(b0names,b0))
    # Add the rest of the parameters to the dictionary
    for (n,v) in zip(name, val)
        if !occursin("b₀", n)
            d[n] = v
        end
    end
    return d
end

function update_df!(df, i, opt)
    if ismissing(opt)
        # If optimization failed, leave df entries as missing
        return
    else
        # Update the dataframe with the optimization results
        df[i, :LL] = opt.lp
        df[i, :iterations] = opt.optim_result.iterations
        df[i, :converged] = ~(any(opt.optim_result.stopped_by) || opt.optim_result.iteration_converged)
        df[i, :runtime_sec] = opt.optim_result.time_run
        # Get dictionary of parameter estimates
        params = get_estimated_params(opt)
        # Update the dataframe with the parameter estimates
        for (param, est) in params
            df[i, "$(param)_est"] = est
        end
    end
end
function update_df_row!(df, opt)
    if ismissing(opt)
        # If optimization failed, leave df entries as missing
        return
    else
        # Update the dataframe with the optimization results
        df[:LL] = opt.lp
        df[:iterations] = opt.optim_result.iterations
        df[:converged] = ~(any(opt.optim_result.stopped_by) || opt.optim_result.iteration_converged)
        df[:runtime_sec] = opt.optim_result.time_run
        # Get dictionary of parameter estimates
        params = get_estimated_params(opt)
        # Update the dataframe with the parameter estimates
        for (param, est) in params
            df["$(param)_est"] = est
        end
    end
end


################################ Simulation Settings ################################
@info "Setting up Simulation."
# Simulation loop settings
S = (;
    Nsigma = 4,  # length(σα² array); Nsigma^2 = number of σα², σμ² grid points
    Nparam = 2,  # Number of true parameter samples, conditional on σα², σμ²
    Nsim = 2,    # Number of simulated datasets per parameter sample
    Nsearch = 4,   # Number of multistart seeds per simulated dataset to find MLE
)

# Create 2D grid for σα², σμ² simulations - log scale so many more points near 0
_x = range(-10, 10, length=S.Nsigma)
XY = [exp10.((σα², σμ²)) for σα² in _x, σμ² in _x ]
# For each σα², σμ² grid point, generate Nparam sets of parameters - b₀, β₁, β₂, ρ, v0
θmat = [param_sampler(σα²σμ²...)() for σα²σμ² in XY, _ in 1:S.Nparam]

# Create a dataframe for all the simulation parameters
df = DataFrame()
for (i, θ) in enumerate(θmat)
    # Flatten the b0 parameter vecotr into 4 separate items
    θ2 = param_flatten(θ)
    for j in 1:S.Nsim, k in 1:S.Nsearch
        θ3 = (; nparam = i, nsim = j, nsearch = k, θ2...)
        push!(df, θ3)
    end
end

# For each parameter column, add an estimate column
nrow = size(df, 1)
for col in names(df)
    if occursin("nsim", col) || occursin("nsearch", col) || occursin("nparam", col)
        continue
    end
    df[!, "$(col)_est"] = Array{Union{Missing, Float64}}(undef, nrow)
end

# Add columns for the optimization results
df[!, :LL] = Array{Union{Missing, Float64}}(undef, nrow)
df[!, :iterations] = Array{Union{Missing, Int64}}(undef, nrow)
df[!, :converged] = Array{Union{Missing, Bool}}(undef, nrow)
df[!, :runtime_sec] = Array{Union{Missing, Float64}}(undef, nrow)

# Optimization settings
maxiter = 100_000
maxtime = 60*60*24  # 24 hours


################################ Simulation Loop ################################
# Define the directory to save temporary files, and create it if it doesn't exist
savedir = "../../data/temp/turing_simulation_output/Nsigma-param-sim-seed_$(S.Nsigma)-$(S.Nparam)-$(S.Nsim)-$(S.Nsearch)"
isdir(savedir) || mkpath(savedir)
@info "\n"^10
@info "Creating save directory:"
@info savedir

# Iterate through each row of the dataframe, simulating the data and estimating the results
Nparams = size(df, 1); n10perc = Nparams//20
println("Staring simulation loop. Will text every 5% of progress ($(round(Float64(n10perc))) iterations).")
@info "Staring simulation loop."
send_txt("Starting Simulation Loop on UCBARE", "0 of Nparams")
# p = Progress(Nparams)
# pmap(1:Nparams) do i
#     # Get the ith row of the dataframe
#     df1 = df[i:i, :]
#     θ = param_vectorize(df1)
#     savefile = "$savedir/simulation_df-$(θ.nparam)-$(θ.nsim)-$(θ.nsearch).csv"

#     # Check if this dataframe already exists
#     if isfile(savefile)
#         df2 = DataFrame(CSV.File(savefile))
#         # Check if optimization has been run and converged
#         if !ismissing(first(df2).converged) && first(df2).converged
#             # If so, skip this row
#             @info "Skipping row $(i) of $(size(df, 1)): $(θ.nparam)-$(θ.nsim)-$(θ.nsearch)"
#             next!(p)
#             continue
#         end
#     end  # If not, run the optimization

#     # Simulate the data
#     Random.seed!(θ.nsim)
#     Y = data_sampler(θ)().Y

#     # Initialize the model with simulated data, generated from θ and random seed
#     Random.seed!(θ.nsearch)
#     model_obj = TuringModels.karp_model5(Y, missing)

#     # Estimate the model
#     opt = try_catch_optim(model_obj; maxiter=maxiter, maxtime)

#     # Save the results
#     update_df!(df1, 1, opt)
#     CSV.write(savefile, df1)

#     # Update the progress bar
#     @info "Finished row $(i) of $(size(df, 1)): $(θ.nparam)-$(θ.nsim)-$(θ.nsearch)"
#     next!(p)

#     # Send text every 5% of iterations
#     (p.counter-1) % n10perc == 0 ? send_txt("Progress: $(p.counter)/$(p.n) = $(round(p.counter/p.n*100, digits=1))%", "") : nothing
# end
# df

simulation_results = @distributed (append!) for i in 1:Nparams
    # Get the ith row of the parameter dataframe
    df1 = deepcopy(df[i:i, :])
    θ = param_vectorize(df1)
    savefile = "$savedir/simulation_df-$(θ.nparam)-$(θ.nsim)-$(θ.nsearch).csv"

    # Check if this dataframe already exists
    if isfile(savefile)
        df2 = DataFrame(CSV.File(savefile))
        # Check if optimization has been run and converged
        if !ismissing(first(df2).converged) && first(df2).converged
            # If so, skip this row
            @info "Skipping row $(i) of $(size(df, 1)): $(θ.nparam)-$(θ.nsim)-$(θ.nsearch)"
            # next!(p)
            continue
        end
    end  # If not, run the optimization

    # Simulate the data
    Random.seed!(θ.nsim)
    Y = data_sampler(θ)().Y

    # Initialize the model with simulated data, generated from θ and random seed
    Random.seed!(θ.nsearch)
    model_obj = TuringModels.karp_model5(Y, missing)

    # Estimate the model
    opt = try_catch_optim(model_obj; maxiter=maxiter, maxtime)

    # Save the results
    update_df!(df1, 1, opt)
    CSV.write(savefile, df1)

    # Update the progress bar
    @info "Finished row $(i) of $(size(df, 1)): $(θ.nparam)-$(θ.nsim)-$(θ.nsearch)"
    # next!(p)
    # # Send text every 5% of iterations
    # (p.counter-1) % n10perc == 0 ? send_txt("Progress: $(p.counter)/$(p.n) = $(round(p.counter/p.n*100, digits=1))%", "") : nothing

    df1
end
@show simulation_results
@info "Finished simulation loop."

# Append the saved dataframes to the main dataframe
for i in 1:size(df, 1)
    df1 = df[i, :]
    θ = param_vectorize(df1)
    savefile = "$savedir/simulation_df-$(θ.nparam)-$(θ.nsim)-$(θ.nsearch).csv"
    df2 = DataFrame(CSV.File(savefile))
    push!(df, df2[1,:])
end
@info "Finished appending saved dataframes."

# Sort by nparam, nsim, nsearch, converged
sort!(df, [:nparam, :nsim, :nsearch, :converged, :LL])
# Drop rows with empty LL (no optimization)
df2 = df[.!ismissing.(df.LL), :]
# Keep uniuqe rows by nparam, nsim, nsearch, LL (in case there are multiple optim results)
df2 = unique(df2, [:nparam, :nsim, :nsearch, :LL])

# Save to file
savefile = "$savedir/simulation_df.csv"
CSV.write(savefile, df2)

@info "Finished saving simulation results to file."
@info "Finished TuringSimulationsParallel.jl"