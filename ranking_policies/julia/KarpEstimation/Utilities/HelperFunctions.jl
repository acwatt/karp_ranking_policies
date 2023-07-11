
module HF
using DataFramesMeta
using CSV
using Dates
# using Statistics

############################################
#               File system functions
############################################
function get_project_root_path(; root_folder_name = "karp_ranking_policies")
    # Search for project root folder by looking for a folder with the root names
    # Assumes that the project root folder is a parent of the current working directory
    dir = dirname(@__DIR__)
    parts = splitpath(dir)
    for i in 1:length(parts)
        if parts[i] == root_folder_name
            return joinpath(parts[1:i]...)
        end
    end
    return nothing
end

ROOT = get_project_root_path()
DATA = joinpath(ROOT, "data")
TEMP = joinpath(DATA, "temp")

############################################
#               Data functions
############################################
create_dataframe() = DataFrame(
    # This function determines the order of the columns
    seed = [], type = [], N = [], T = [],
    ρ_estimate = [], ρ_start = [], ρ_lower = [], ρ_upper = [],
    σₐ²_estimate = [],  σₐ²_start = [], σₐ²_lower = [], σₐ²_upper = [],
    σᵤ²_estimate = [],  σᵤ²_start = [],  σᵤ²_lower = [], σᵤ²_upper = [], 
    iterations = [], LL = [],
    V_norm_finalstep = [], V_norm_true=[], V = [],
    b₀₁ = [], b₀₂ = [], b₀₃ = [], b₀₄ = [], 
    β₁ = [], β₂ = [], method = [],
    runtime = [], notes=[]
)


"""If missing, return missing n-vector.
    Otherwise, return rounded var vector with missing appended to end to 
    fill out to n-vector.
"""
function optional_round(var, n, digits)
    if ismissing(var)
        if n == 1
            return missing
        else
            return repeat([missing], n)
        end
    else
        var = round.(var, digits=digits)
        if length(var) < n
            return vcat(var, repeat([missing], n - length(var)))
        end
        return var
    end
end


function add_row!(df, seed, type, N, T;
    b₀ᵢ = missing, β = missing,
    ρ=missing, σₐ²=missing, σᵤ²=missing, iterations=missing,
    LL=missing, V_norm_finalstep=missing, V_norm_true=missing, V=missing,
    params_start = missing, params_lower = missing, params_upper = missing,
    runtime=missing, notes=missing, digits=3, method = missing)

    if N == 2; b₀₁, b₀₂ = optional_round(b₀ᵢ, N, digits); b₀₃, b₀₄ = missing, missing;
    elseif N == 3; b₀₁, b₀₂, b₀₃ = optional_round(b₀ᵢ, N, digits); b₀₃ = missing;
    elseif N == 4; b₀₁, b₀₂, b₀₃, b₀₄ = optional_round(b₀ᵢ, N, digits);
    end
    β₁, β₂ = optional_round(β, 2, digits)

    ρ = optional_round(ρ, 1, 4)
    println(ρ)
    σₐ² = optional_round(σₐ², 1, digits+4)
    σᵤ² = optional_round(σᵤ², 1, digits+4)

    LL = optional_round(LL, 1, 4)
    V_norm_finalstep = optional_round(V_norm_finalstep, 1, digits)
    V_norm_true = optional_round(V_norm_true, 1, digits)

    ρ_start, σₐ²_start, σᵤ²_start = optional_round(params_start, 3, 4)
    ρ_lower, σₐ²_lower, σᵤ²_lower = optional_round(params_lower, 3, 4)
    ρ_upper, σₐ²_upper, σᵤ²_upper = optional_round(params_upper, 3, 4)

    push!(df, Dict(
    :seed=>seed, :type=>type, :N=>N, :T=>T,
    :b₀₁=>b₀₁, :b₀₂=>b₀₂, :b₀₃=>b₀₃, :b₀₄=>b₀₄,
    :β₁=>β₁,   :β₂=>β₂,
    :ρ_estimate=>ρ,    :σₐ²_estimate=>σₐ²,    :σᵤ²_estimate=>σᵤ², :LL=>LL, :iterations=>iterations,
    :V_norm_finalstep=>V_norm_finalstep,      :V_norm_true=>V_norm_true, :V=>V,
    :ρ_start=>ρ_start, :σₐ²_start=>σₐ²_start, :σᵤ²_start=>σᵤ²_start, 
    :ρ_lower=>ρ_lower, :σₐ²_lower=>σₐ²_lower, :σᵤ²_lower=>σᵤ²_lower, 
    :ρ_upper=>ρ_upper, :σₐ²_upper=>σₐ²_upper, :σᵤ²_upper=>σᵤ²_upper, 
    :method=>method,   :runtime=>runtime,     :notes=>notes
    ))
end


create_simulation_results_df() = DataFrame(
    # This function determines the order of the columns
    seed = Int64[], N = Int64[], T = Int64[],
    ρ_true = Float64[],   ρ_est = Float64[],    ρ_start = Float64[],    ρ_lower = Float64[],   ρ_upper = Float64[],
    σₐ²_true = Float64[], σₐ²_est = Float64[],  σₐ²_start = Float64[],  σₐ²_lower = Float64[], σₐ²_upper = Float64[],
    σᵤ²_true = Float64[], σᵤ²_est = Float64[],  σᵤ²_start = Float64[],  σᵤ²_lower = Float64[], σᵤ²_upper = Float64[],
    LL = Float64[], method=[], runtime = [], notes=[]
)

function summarize_simulation_results(df)
    df1 = select(df, Not([:seed,:σₐ²_est, :σᵤ²_est, :LL, :method]))[1:1,:]
    df2 = @chain df begin
        @transform(:σₐ²_diff = abs.(:σₐ²_est - :σₐ²_true), :σᵤ²_diff = abs.(:σᵤ²_est - :σᵤ²_true))
        @by(:method, 
            :σₐ²_diff_mean = mean(:σₐ²_diff), :σₐ²_diff_std = std(:σₐ²_diff), 
            :σᵤ²_diff_mean = mean(:σᵤ²_diff), :σᵤ²_diff_std = std(:σᵤ²_diff),
            :LL_mean = mean(:LL), :LL_std = std(:LL), :seeds = length(unique(:seed))
        )
        @transform(:N = unique(df.N)[1], :T = unique(df.T)[1])
        leftjoin(df1, on=[:N,:T])
    end
    return df2
end

function add_row_simulation_results!(df, seed, type, N, T;
        θhat = missing, θ₀ = missing, θLB = missing, θUB = missing,
        LL=missing, runtime=missing, notes=missing, method = missing)
    push!(df, (
        seed=seed, type=type, N=N, T=T,
        ρ_estimate=θhat.ρ,    σₐ²_estimate=θhat.σₐ²,    σᵤ²_estimate=θhat.σᵤ², LL=LL,
        ρ_start=θ₀.ρ, σₐ²_start=θ₀.σₐ², σᵤ²_start=θ₀.σᵤ², 
        ρ_lower=θLB.ρ, σₐ²_lower=θLB.σₐ², σᵤ²_lower=θLB.σᵤ², 
        ρ_upper=θUB.ρ, σₐ²_upper=θUB.σₐ², σᵤ²_upper=θUB.σᵤ², 
        method=method,   runtime=runtime,     notes=notes
    ))
end


function write_simulation_df(df; N=2)
    # Save dataframe
    filepath = joinpath(TEMP, "simulation_results_log(N=$N).csv")
    if isfile(filepath)
        CSV.write(filepath, df,append=true)
    else
        CSV.write(filepath, df)
    end
end


function write_estimation_df(df; N=2)
    # Round dataframe columns
    for n in names(df)
        if eltype(df[!,n]) == Float64 || eltype(df[!,n]) == Float32
            df[!,n] = round.(df[!,n], digits=3)
        end
    end
    # Save dataframe
    filepath = joinpath(TEMP, "realdata_estimation_results_log.csv")
    if isfile(filepath)
        CSV.write(filepath, df, append=true)
    else
        CSV.write(filepath, df)
    end
end


function write_estimation_df_notrend(df, search_start; suffix="")
    # Save dataframe
    filepath = joinpath(TEMP, "simulation_estimation$(suffix)_results_log_start$search_start.csv")
    if isfile(filepath)
        CSV.write(filepath, df, append=true)
    else
        CSV.write(filepath, df)
    end
end

function read_estimation_df_notrend(search_start)
    filepath = joinpath(TEMP, "simulation_estimation_results_log_start$search_start.csv")
    df = CSV.read(filepath, DataFrame)
end


function write_data_df(df, data_type; N=4)
    @transform!(df, :type = data_type)
    @transform!(df, :N = N)
    # Save dataframe
    filepath = joinpath(TEMP, "realdata_estimation_data.csv")
    if isfile(filepath)
        CSV.write(filepath, df, append=true)
    else
        CSV.write(filepath, df)
    end
end


function read_data(N, T)
    filepath = joinpath(DATA, "sunny", "clean", "grouped_allYears_nation.1751_2014.csv")
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


function read_global_data(T)
    filepath = joinpath(DATA, "sunny", "clean", "ts_allYears_nation.1751_2014.csv")
    df = @chain CSV.read(filepath, DataFrame) begin
        @rsubset :Year >= 1945 && :Year < 1945+T && :group ∈ ["USA","EU","BRIC","Other"][1:N]
        @transform(:group = get.(Ref(country2num), :group, missing))
        @select(:t = :time .- 44,
                :i = categorical(:group),
                :eᵢₜ = :ebar)
        @orderby(:t, :i)
    end

    return(df)
end


function get_covar_names(data)
    ilevels = unique(data.i); len = length(ilevels);
    inames = repeat(["i"], len-1) .* string.(ilevels[2:len])
    varnames = vcat(["intercept"], inames, ["t", "t^2"])
    return(varnames)
end


function get_sample_variances(data)
    σₐ² = sum((data.αₜ .- mean(data.αₜ)) .^ 2) / 2 / (T - 1)
    σᵤ² = sum((data.μᵢₜ .- mean(data.μᵢₜ)) .^ 2) / (N * T - 1)
    return(σₐ², σᵤ²)
end


function save_reformatted_data()
    N=2; T=60;
    data = read_data(N, T)
    filepath = joinpath(TEMP, "realdata_reformated(N=$N).csv")
    CSV.write(filepath, data)
end


function translate_gls(gls, N)
    β = gls[:β][N+1:N+2]
    b₀ᵢ = gls[:β][1:N]
    for i ∈ 2:N
        b₀ᵢ[i] = gls[:β][1] + gls[:β][i]
    end
    return([β, b₀ᵢ])
end


function save_ols!(df::DataFrame, gls::Dict, N, T; data_type = "real data")
    β, b₀ᵢ = translate_gls(gls, N)
    note = "This is the first-pass OLS estimate of b₀ᵢ and β using real data (using GLS with the identity matrix)."
    add_row!(df, data_type, "First-pass OLS Estimate", N, T;
        b₀ᵢ=b₀ᵢ, β=β, iterations=0,
        runtime=Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
        notes=note
    )
end


function save_simulated_params!(df::DataFrame, params, N, T; data_type = "simulated data")
    β = params[:β]
    b₀ᵢ = params[:b₀ᵢ]
    note = "b₀ᵢ and β used to generate simulated data"
    add_row!(df, data_type, "Parameters in simulation", N, T;
        b₀ᵢ=b₀ᵢ, β=β, iterations=0,
        runtime=Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
        notes=note
    )
end

function get_results_σstarts(date)
    filepath = joinpath(TEMP, "realdata_estimation_results_log_$date.csv")
    df = @chain CSV.read(filepath, DataFrame,
            select=["type", "V_norm_finalstep", "??\xb2_start", "??\xb2_start_1"],
            footerskip=184) begin
        @rsubset :V_norm_finalstep == 0
        @select(:αstart=cols(Symbol("??\xb2_start")),
                :μstart=cols(Symbol("??\xb2_start_1")))
    end
    return(df)
end


param(plist) = Dict(
    :ρ => plist[1],
    :σₐ² => plist[2],
    :σᵤ² => plist[3],
)

end