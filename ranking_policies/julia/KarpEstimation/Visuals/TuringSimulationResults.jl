# using KarpEstimation environment
using DataFrames, DataFramesMeta, CSV
using StatsPlots
using Statistics
using PlotlyJS

# Load "C:\Users\A\github\karp_ranking_policies\data\temp\turing_simulation_output\Nsigma-param-sim-seed_5-5-5-5\simulation_df.csv" into dataframe
sim_results_file = "C:\\Users\\A\\github\\karp_ranking_policies\\data\\temp\\turing_simulation_output\\Nsigma-param-sim-seed_5-5-5-5\\simulation_df.csv"
df = DataFrame(CSV.File(sim_results_file))

# Calculate bias and absolute bias of each parameter where bias = param_est - param
params = names(df)[4:13]
for param in params
    df[!, param * "_bias"] = df[!, param * "_est"] - df[!, param]
    df[!, param * "_biasperc"] = df[!, param * "_bias"] ./ df[!, param]
    df[!, param * "_abias"] = abs.(df[!, param * "_bias"])
    df[!, param * "_abiasperc"] = abs.(df[!, param * "_biasperc"])
end


################################### Plot 1 - basic ###################################
# Group df by σα², σμ²
df1 = @chain df begin
    @select(:σα², :σμ², :σα²_bias, 
            :σμ²_bias, :σα²_abias, :σμ²_abias, 
            :σα²_biasperc, :σμ²_biasperc, :σα²_abiasperc, :σμ²_abiasperc)
    @by([:σμ², :σα²], 
        :σα²_bias_avg = mean(:σα²_bias), :σμ²_bias_avg = mean(:σμ²_bias), 
        :σα²_abias_avg = mean(:σα²_abias), :σμ²_abias_avg = mean(:σμ²_abias),
        :σα²_biasperc_avg = mean(:σα²_biasperc), :σμ²_biasperc_avg = mean(:σμ²_biasperc),
        :σα²_abiasperc_avg = mean(:σα²_abiasperc), :σμ²_abiasperc_avg = mean(:σμ²_abiasperc))
    @transform(:log_σα² = log10.(:σα²), :log_σμ² = log10.(:σμ²))
    @transform(:log_σμ²_abiasperc = log10.(:σα²_abiasperc_avg))
end
# Save df1 to csv
CSV.write("C:\\Users\\A\\github\\karp_ranking_policies\\data\\temp\\turing_simulation_output\\Nsigma-param-sim-seed_5-5-5-5\\simulation_df1.csv", df1)
# Make a heatmap of the bias of σα² and σμ² using log_σα² and log_σμ²
PlotlyJS.plot(
    PlotlyJS.heatmap(df1,
        x=:log_σα², y=:log_σμ², z=:σα²_bias_avg, title="Avg Bias of σα²")
)
# Make a heatmap of the absolute bias of σα² and σμ² using log_σα² and log_σμ²
PlotlyJS.plot(
    PlotlyJS.heatmap(df1,
        x=:log_σα², y=:log_σμ², z=:σα²_abias_avg, title="Avg Absolute Bias of σα²")
)
# Make a heatmap of the bias percentage of σμ² and σα² using log_σμ² and log_σα²
PlotlyJS.plot(
    PlotlyJS.heatmap(df1,
        x=:log_σμ², y=:log_σα², z=:σα²_biasperc_avg, title="Avg Bias Percentage of σα²")
)
# Make a heatmap of the absolute bias percentage of σμ² and σα² using log_σμ² and log_σα²
PlotlyJS.plot(
    PlotlyJS.heatmap(df1,
        x=:log_σμ², y=:log_σα², z=:σα²_abiasperc_avg, title="Avg Absolute Bias Percentage of σα²")
)
# Make a heatmap of the absolute bias percentage of σμ² and σα² using log_σμ² and log_σα²
PlotlyJS.plot(
    PlotlyJS.heatmap(df1,
        x=:log_σμ², y=:log_σα², z=:log_σμ²_abiasperc, title="Avg Absolute Bias Percentage of σα²")
)

# requires z to be rectangular
# make x and y arrays and z a matrix
x = unique(df1.log_σα²)
y = unique(df1.log_σμ²)
z = reshape(df1.σα²_bias_avg, length(x), length(y))
StatsPlots.heatmap(x,y,z, title="Avg Bias of σα²")

################################### Plot 2 - restrict to ρ<1 ###################################
# Group df by σα², σμ²
df2 = @chain df begin
    @subset(:ρ .< 1)
    @select(:σα², :σμ², :σα²_bias, 
            :σμ²_bias, :σα²_abias, :σμ²_abias, 
            :σα²_biasperc, :σμ²_biasperc, :σα²_abiasperc, :σμ²_abiasperc)
    @by([:σμ², :σα²], 
        :σα²_bias_avg = mean(:σα²_bias), :σμ²_bias_avg = mean(:σμ²_bias), 
        :σα²_abias_avg = mean(:σα²_abias), :σμ²_abias_avg = mean(:σμ²_abias),
        :σα²_biasperc_avg = mean(:σα²_biasperc), :σμ²_biasperc_avg = mean(:σμ²_biasperc),
        :σα²_abiasperc_avg = mean(:σα²_abiasperc), :σμ²_abiasperc_avg = mean(:σμ²_abiasperc))
    @transform(:log_σα² = log10.(:σα²), :log_σμ² = log10.(:σμ²))
end




# Group df by σα², σμ²
df2 = @chain df begin
    @subset(:ρ < 1)
    select(:σα², :σμ², :σα²_bias, :σμ²_bias, :σα²_abias, :σμ²_abias)
    groupby(:σα², :σμ²)
    @combine mean
    @transform($"Top Rate" = parse.(Float64, strip.(:Rates,'%')),
        $"State" = states)
    @select($"State", $"Top Rate")
    # Merge population onto tax rates
    leftjoin(df2, on=:State)
    sort(["Top Rate", "pop"], rev=true)
    @select($"State", $"Top Rate", $"Population")
    dropmissing(:Rates)
    @subset(:State .!= "D.C.")
end