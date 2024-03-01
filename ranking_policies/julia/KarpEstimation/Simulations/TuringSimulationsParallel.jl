# This file is designed to create highly parallel simuluations and estimation of the model using Turing.

println(@__DIR__)
using Pkg; Pkg.activate(joinpath(@__DIR__, "turingSimulations"))
# ]add Turing DataFrames Optim ProgressMeter Dates CSV NamedArrays LoggingExtras SMTPClient
using Turing
using DataFrames, DataFramesMeta
using Dates
# using LinearAlgebra
using Optim
# using StatsBase  # for coeftable and stderror
# using StatsPlots  # for histogram
using ProgressMeter
using Random
using Logging, LoggingExtras
using CSV

include("../Model/TuringModels.jl")  # TuringModels
include("../Utilities/Communications.jl")  # send_txt


################################ logging ################################
isdir("output/logs") || mkpath("output/logs")
LOGFILE = "output/logs/logfile-$(Dates.format(Dates.now(), "yyyy-mm-dd")).txt"
# LOGFILE = "output/logs/logfile.txt"
if isa(global_logger(), ConsoleLogger)
    OLDLOGGER = global_logger()
end
FILELOGGER = FormatLogger(LOGFILE; append = true) do io, args
    println(io, "[", args.level, " ", Dates.now(), "] ", args.message)
end
global_logger(FILELOGGER)
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
# Simulation loop settings
S = (;
    Nsigma = 10,  # length(σα² array); Nsigma^2 = number of σα², σμ² grid points
    Nparam = 10,  # Number of true parameter samples, conditional on σα², σμ²
    Nsim = 10,    # Number of simulated datasets per parameter sample
    Nsearch = 10,   # Number of multistart seeds per simulated dataset to find MLE
)

# Create 2D grid for σα², σμ² simulations - log scale so many more points near 0
_x = range(-10, 10, length=S.Nsigma)
xx = exp10.(_x)
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
df

# Optimization settings
maxiter = 100_000
maxtime = 60*60*24  # 24 hours


################################ Simulation Loop ################################
# Define the directory to save temporary files, and create it if it doesn't exist
savedir = "../../data/temp/turing_simulation_output/Nsigma-param-sim-seed_$(S.Nsigma)-$(S.Nparam)-$(S.Nsim)-$(S.Nsearch)"
isdir(savedir) || mkpath(savedir)

# # Print dataframe row in ereadable format
# i_ = 1
# ["$n = $(df[i_,n])" for n in names(df[i_,:])]


# Iterate through each row of the dataframe, simulating the data and estimating the results
Nparams = size(df, 1); n10perc = Nparams//20
println("Staring simulation loop. Will text every 5% of progress ($(round(Float64(n10perc))) iterations).")
send_txt("Starting Simulation Loop on UCBARE", "")
p = Progress(Nparams)
Threads.@threads for i in 1:size(df, 1)
    df1 = df[i:i, :]
    # Get the ith row of the dataframe
    θ = param_vectorize(df1)
    savefile = "$savedir/simulation_df-$(θ.nparam)-$(θ.nsim)-$(θ.nsearch).csv"
    # Check if this dataframe already exists
    if isfile(savefile)
        df2 = DataFrame(CSV.File(savefile))
        # Check if optimization has been run and converged
        if !ismissing(first(df2).converged) && first(df2).converged
            # If so, skip this row
            @info "Skipping row $(i) of $(size(df, 1)): $(θ.nparam)-$(θ.nsim)-$(θ.nsearch)"
            println("Skipping row $(i) of $(size(df, 1)): $(θ.nparam)-$(θ.nsim)-$(θ.nsearch)")
            next!(p)
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
    next!(p)
    # Send text every 5% of iterations
    (p.counter-1) % n10perc == 0 ? send_txt("Progress: $(p.counter)/$(p.n) = $(round(p.counter/p.n*100, digits=1))%", "") : nothing
end
df


# Append the saved dataframes to the main dataframe
for i in 1:size(df, 1)
    df1 = df[i, :]
    θ = param_vectorize(df1)
    savefile = "$savedir/simulation_df-$(θ.nparam)-$(θ.nsim)-$(θ.nsearch).csv"
    df2 = DataFrame(CSV.File(savefile))
    push!(df, df2[1,:])
end

# Sort by nparam, nsim, nsearch, converged
sort!(df, [:nparam, :nsim, :nsearch, :converged, :LL])
# Drop rows with empty LL (no optimization)
df2 = df[.!ismissing.(df.LL), :]
# Keep uniuqe rows by nparam, nsim, nsearch, LL (in case there are multiple optim results)
df2 = unique(df2, [:nparam, :nsim, :nsearch, :LL])

# Save to file
savefile = "$savedir/simulation_df.csv"
CSV.write(savefile, df2)

global_logger(OLDLOGGER);



################################ Analysis of Simulation Data ################################
#! Start here
# Switch environments to use Gadfly
using Pkg
env_path = joinpath(splitpath(@__DIR__)[1:end-1]..., "Utilities", "test")
Pkg.activate(env_path)
Pkg.add(["Gadfly", "DataFrames", "DataFramesMeta", "CSV", "Statistics", "ColorSchemes", "Colors"])
using Gadfly
using ColorSchemes, Colors
using ColorSchemeTools
using Compose
using DataFrames, DataFramesMeta
using CSV
using Statistics

# Copied settings from above simulations
S = (;
    Nsigma = 10,  # length(σα² array); Nsigma^2 = number of σα², σμ² grid points
    Nparam = 10,  # Number of true parameter samples, conditional on σα², σμ²
    Nsim = 10,    # Number of simulated datasets per parameter sample
    Nsearch = 10,   # Number of multistart seeds per simulated dataset to find MLE
)

# Create 2D grid for σα², σμ² simulations - log scale so many more points near 0
_x = range(-10, 10, length=S.Nsigma)
xx = exp10.(_x)

# Load the simulation data
savedir = "../../data/temp/turing_simulation_output/Nsigma-param-sim-seed_$(S.Nsigma)-$(S.Nparam)-$(S.Nsim)-$(S.Nsearch)"
savedir = joinpath(splitpath(@__DIR__)[1:end-4]..., "data", "temp", "turing_simulation_output", "Nsigma-param-sim-seed_$(S.Nsigma)-$(S.Nparam)-$(S.Nsim)-$(S.Nsearch)")
isdir(savedir)
savefile = "$savedir/simulation_df.csv"
df = DataFrame(CSV.File(savefile))

############ Bias of Parameter Estimates ############
params = [:b₀₁, :b₀₂, :b₀₃, :b₀₄, :β₁, :β₂, :σα², :σμ², :ρ, :v0]

for param in params
    df = @transform(df, $"$(param)_est_bias" = $"$(param)_est" - $"$(param)")
end

############ Summary Statistics of Parameter Estimates ############
# Select search with largest LL for each nparam, nsim
df_best = @chain df begin
    groupby([:nparam, :nsim])
    @subset(:LL .== maximum(:LL))
    groupby([:nparam, :nsim])
    @subset(:iterations .== minimum(:iterations))
end
df_best = unique(df_best, [:nparam, :nsim])


# Get the bias and standard error of the parameter estimates
df_avg = @chain df_best begin
    groupby([:σα², :σμ²])
    @combine(
        :σα²_est_mean = mean(:σα²_est),
        :σα²_bias_mean = mean(:σα²_est_bias),
        :σα²_absbias_mean = mean(abs.(:σα²_est_bias)),
        :σα²_abspercbias_mean = mean(abs.(:σα²_est_bias / :σα²)),
        :σα²_bias_std = std(:σα²_est_bias),
        :σα²_bias_min = minimum(:σα²_est_bias),
        :σα²_bias_max = maximum(:σα²_est_bias),
        :σμ²_est_mean = mean(:σμ²_est),
        :σμ²_bias_mean = mean(:σμ²_est_bias),
        :σμ²_absbias_mean = mean(abs.(:σμ²_est_bias)),
        :σμ²_abspercbias_mean = mean(abs.(:σμ²_est_bias / :σμ²)),
        :σμ²_bias_std = std(:σμ²_est_bias),
        :σμ²_bias_min = minimum(:σμ²_est_bias),
        :σμ²_bias_max = maximum(:σμ²_est_bias),
    )
end




# Drop all rows with ρ=1, and calculate bias and standard error of the parameter estimates
sum(df_best.ρ_est .== 1) / nrow(df_best)
df_avg_restricted = @chain df_best begin
    @subset(:ρ_est .!= 1)
    groupby([:σα², :σμ²])
    @combine(
        :count = length(:σα²),
        :σα²_est_mean = mean(:σα²_est),
        :σα²_bias_mean = mean(:σα²_est_bias),
        :σα²_percbias_mean = mean(:σα²_est_bias / :σα²),
        :σα²_absbias_mean = mean(abs.(:σα²_est_bias)),
        :σα²_abspercbias_mean = mean(abs.(:σα²_est_bias / :σα²)),
        :σα²_bias_std = std(:σα²_est_bias),
        :σα²_bias_min = minimum(:σα²_est_bias),
        :σα²_bias_max = maximum(:σα²_est_bias),
        :σμ²_est_mean = mean(:σμ²_est),
        :σμ²_bias_mean = mean(:σμ²_est_bias),
        :σμ²_percbias_mean = mean(:σμ²_est_bias / :σμ²),
        :σμ²_absbias_mean = mean(abs.(:σμ²_est_bias)),
        :σμ²_abspercbias_mean = mean(abs.(:σμ²_est_bias / :σμ²)),
        :σμ²_bias_std = std(:σμ²_est_bias),
        :σμ²_bias_min = minimum(:σμ²_est_bias),
        :σμ²_bias_max = maximum(:σμ²_est_bias),
    )
end





# Restrict to ρ < 1 before selecting max LL and min iterations
df_best_restricted = @chain df begin
    @subset(:ρ_est .!= 1)
    groupby([:nparam, :nsim])
    @subset(:LL .== maximum(:LL))
    groupby([:nparam, :nsim])
    @subset(:iterations .== minimum(:iterations))
end
df_best_restricted = unique(df_best_restricted, [:nparam, :nsim])

# Get the bias and standard error of the parameter estimates, restricted to ρ < 1
df_avg_restricted = @chain df_best_restricted begin
    groupby([:σα², :σμ²])
    @combine(
        :count = length(:σα²),
        :σα²_est_mean = mean(:σα²_est),
        :σα²_bias_mean = mean(:σα²_est_bias),
        :σα²_absbias_mean = mean(abs.(:σα²_est_bias)),
        :σα²_abspercbias_mean = mean(abs.(:σα²_est_bias / :σα²)),
        :σα²_bias_std = std(:σα²_est_bias),
        :σα²_bias_min = minimum(:σα²_est_bias),
        :σα²_bias_max = maximum(:σα²_est_bias),
        :σμ²_est_mean = mean(:σμ²_est),
        :σμ²_bias_mean = mean(:σμ²_est_bias),
        :σμ²_absbias_mean = mean(abs.(:σμ²_est_bias)),
        :σμ²_abspercbias_mean = mean(abs.(:σμ²_est_bias / :σμ²)),
        :σμ²_bias_std = std(:σμ²_est_bias),
        :σμ²_bias_min = minimum(:σμ²_est_bias),
        :σμ²_bias_max = maximum(:σμ²_est_bias),
    )
end


# Select nsim datasets that get smallest σα²
df_smallest = @chain df begin
    # Filter to best estimate for each simulated dataset (Get MLE with maximim LL from the same simulated data with different starting search parameters)
    groupby([:nparam, :nsim])
    @subset(:LL .== maximum(:LL))
    # Filter to smallest σα² est of all datasets from each true σα², σμ² pair 
    groupby([:σα², :σμ²])
    @subset(:σα²_est .== minimum(:σα²_est))
    # If any duplicates, select largest LL
    groupby([:σα², :σμ²])
    @subset(:LL .== maximum(:LL))
end





################### Gadfly ###################



# visualize data
set_default_plot_size(17cm, 14cm)
z_minmax = extrema(log10.(df_avg[!, :σα²_est_mean]))

cs1 = ColorScheme(reverse(Colors.sequential_palette(300, 100, logscale=true)))

terrain_data = (
        (0.00, (0.2, 0.2, 0.6)),
        (0.15, (0.0, 0.6, 1.0)),
        (0.25, (0.0, 0.8, 0.4)),
        (0.50, (1.0, 1.0, 0.6)),
        (0.75, (0.5, 0.36, 0.33)),
        (1.00, (1.0, 1.0, 1.0)))
terrain = make_colorscheme(terrain_data, length = 50)
d = (
        (z_minmax[1], (0.2, 0.2, 0.6)),
        (0, (1, 1, 1)),
        (z_minmax[2], (0.0, 0.8, 0.4)))
t = make_colorscheme(d, length = 50)

rng = z_minmax[2] - z_minmax[1]
zero_fraction = abs(z_minmax[1]) / rng

zmax = 150
zmin = -50
zero_fraction = abs(zmin) / (zmax - zmin)
# zero_fraction = 0.2
dx_l = zero_fraction/10
dx_r = (1-zero_fraction)/100
least_intensity = 0.9
most_intensity = 0.4


cdict = Dict(:blue   => ((0.0,  most_intensity,  most_intensity),
                        (zero_fraction,  least_intensity,  1.0),
                        (zero_fraction+dx_r, 0,  0),
                        (1.0,  0,  0)),
             :green => ((0.0,  0.0,  0.0),
                        (zero_fraction-dx_l, 0,  0),
                        (zero_fraction, 1,  1),
                        (zero_fraction+dx_r, 0,  0),
                        (1.0,  0,  0)),
             :red  => ((0.0,  0.0,  0.0),
                        (zero_fraction-dx_l, 0,  0),
                        (zero_fraction, 1,  least_intensity),
                        (1.0,  most_intensity,  most_intensity))
)
divergent_scheme = make_colorscheme(cdict; length=1000)

function mylog(x)
    if x == 0
        return -100
    elseif log10(x) < -100
        return -100
    elseif log10(x) > 100
        return 100
    else
        return log10(x)
    end
end

# 
function plot_heatmap(df, zcol)
    df.z = mylog.(df[!, zcol])
    # z_minmax = extrema(df.z)
    # z_absmax = minimum(abs.(z_minmax))
    # Truncate z values to either negative or positive max, whichever is smaller, to make symmetrical color scale
    # z = max.(z, -z_absmax)
    # z = min.(z, z_absmax)
    p = Gadfly.plot(
        df,
        x = :σα²,
        y = :σμ²,
        color = :z,
        Scale.x_log10, Scale.y_log10,
        Geom.rectbin,
        Scale.ContinuousColorScale(
            palette -> get(divergent_scheme, palette)
            # vlag 
        ),
        # Guide.xticks( # Don't use this -- it used all the memory on the server last time
        #     ticks=[minimum(df.σα²):maximum(df.σα²);]
        # ),
        # Set axis limits
        Coord.cartesian(xmin=-10.5, xmax=10.5, ymin=-10.5, ymax=10.5),
        # set font size for colorbar, and background color
        Theme(background_color = "white", key_label_font_size=10pt, key_title_font_size=10pt),
        # Colorbar title
        Guide.colorkey(title="log10($zcol)"),
        # Add annotations
        Guide.annotation(
            compose(
                context(),
                text(
                    log10.(df.σα²),  # x var
                    log10.(df.σμ²),  # y var
                    string.(round.(df.z, digits=2)),  # z var
                    repeat([hcenter], nrow(df)),  # dataframe
                    repeat([vcenter], nrow(df)),
                ),
                fontsize(7pt),
                stroke("black"),
            ),
        )
    )
    return p
end

# p = plot_heatmap(df_avg, :σα²_bias_mean)
# p = plot_heatmap(df_avg_restricted, :σα²_est_mean)
p2 = plot_heatmap(df_smallest, :σα²_est)





# Build rectangular z array for heatmap
function build_logz(df, param, stat_suffix="bias_mean")
    z = zeros(S.Nsigma, S.Nsigma)
    for i in 1:S.Nsigma, j in 1:S.Nsigma
        σα² = xx[j]  # x (columns)
        σμ² = xx[i]  # y (rows)
        z[i,j] = @chain df begin
            @subset(:σα² .== σα²)
            @subset(:σμ² .== σμ²)
            @select($"$(param)_$(stat_suffix)")
            mylog(first(_)[1])
        end
    end
    return z
end

σα²_bias_mean = build_z(df_avg, :σα², "bias_mean")
σα²_abspercbias_mean = build_z(df_avg, :σα², "abspercbias_mean")
σα²_est_mean = build_logz(df_avg_restricted, :σα², "est_mean")
σα²_est_small = build_logz(df_smallest, :σα², "est")














############ Plotting with Plots (doesn't work on server) ############


using Plots
gr()
heatmap(xx, xx, σα²_bias_mean; xscale=:log10, yscale=:log10, title="Mean σα² Bias")
Plots.heatmap(randn(10,10))

heatmap(xx, xx, mean(results_nt.σα².mean_se, dims=3)[:,:,1]; xscale=:log10, yscale=:log10, title="Mean σα² Standard Error")
heatmap(xx, xx, mean(results_nt.σα².mean_bias, dims=3)[:,:,1]; xscale=:log10, yscale=:log10, title="Mean σα² Bias")
heatmap(xx, xx, mean(results_nt.σα².mean_abs_bias, dims=3)[:,:,1]; xscale=:log10, yscale=:log10, title="Mean σα² Absolute Bias")



using Plots
gr()
data = rand(21,100)
heatmap(1:size(data,1),
    1:size(data,2), data,
    c=cgrad([:blue, :white,:red, :yellow]),
    xlabel="x values", ylabel="y values",
    title="My title"
)

using Pkg
pkg"add GR_jll"
using GR_jll
Pkg.status()


ENV["PYTHON"]=""
Pkg.add("PyCall")
Pkg.build("PyCall")

using PyPlot
# use x = linspace(0,2*pi,1000) in Julia 0.6
x = range(0; stop=2*pi, length=1000); y = sin.(3 * x + 4 * cos.(2 * x));
plot(x, y, color="red", linewidth=2.0, linestyle="--")
title("A sinusoidally modulated sinusoid")

#################### matplotlib ####################
import seaborn as sns
import matplotlib.pylab as plt

uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data, linewidth=0.5)
plt.show()
