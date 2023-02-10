using Plots
using ProgressMeter
using CSV, Tables
pyplot()
function plot_ll_big_sigmas()
    # import Pkg; Pkg.add("PlotlyJS")


    # get some initial residuals
    σₐ² = σᵤ² = 10^11;
    ρ = starting_params[:ρ];    β = starting_params[:β]
    σₐ² = starting_params[:σₐ²];  σᵤ² = starting_params[:σᵤ²]
    N = 4; T = 60; ρ = 0.8785;
    # data = read_data(N, T)
    data = dgp(ρ, σₐ², σᵤ², β, N, T; random_seed = 1)
    V = Diagonal(ones(length(data.eᵢₜ)))
    gls = mygls(@formula(eᵢₜ ~ 1 + i + t + t^2), data, V, N)
    v = vec(gls[:resid])


    # create log-likelihood function of just σₐ², σᵤ²
    function LL10_plot(σₐ², σᵤ²)
        # println("σₐ² = ", σₐ², "    σᵤ² = ", σᵤ²)
        return -log(10, nLL(ρ, σₐ², σᵤ², v, N, T))
    end
    function LL_plot(σₐ², σᵤ²)
        # println("σₐ² = ", σₐ², "    σᵤ² = ", σᵤ²)
        return -nLL(ρ, σₐ², σᵤ², v, N, T)
    end
    # println("β: ", gls[:β])
    # println(v)

    # Evaluate LL at σₐ² = 1, σᵤ² = 10
    # println(LL_plot(1,10))
    # println(LL_plot(5e4^2,5e4^2))

    # σₐ²vec = 2e8:5e10:1e12
    # σᵤ²vec = 2e12:2e12:6e11

    σₐ²vec = 10 .^(-1:0.1:1)
    σᵤ²vec = 10 .^(10.54:0.002:10.57)
    σᵤ²bar = 10^10.56
    LLvect = @showprogress [LL_plot(a,σᵤ²bar) for a in σₐ²vec]
    p4 = plot(σₐ²vec, LLvect,
        xlabel="σα²",
        ylabel="LL",
        title="Log Likelihood")
    p5 = plot(σₐ²vec, LLvect,
        xlabel="σα²", xscale=:log,
        ylabel="LL",
        title="Log Likelihood")
    p6 = plot(σₐ²vec, LLvect,
        xlabel="σα²", xscale=:log,
        ylabel="LL", yscale=:log,
        title="Log Likelihood")
    
    #! I wasn't able to get most of these to save to file in the REPL,
    #! so I had to take screenshots of the plotting window.


    fdsa
    X = repeat(reshape(σₐ²vec, 1, :), length(σᵤ²vec), 1)
    Y = repeat(σᵤ²vec, 1, length(σₐ²vec))
    Z = @showprogress map(LL10_plot, X, Y)

    mat = [σᵤ²vec zeros(size(σᵤ²vec)) Z]
    header = [0 0 σₐ²vec']
    mat = [header; zeros(size(header)); mat]

    
    filepath = "../../data/temp/LL real_data N$N T$T ρ$ρ.csv"
    CSV.write(filepath,  Tables.table(mat), writeheader=false)

    p3 = surface(σₐ²vec, σᵤ²vec, Z, xscale = :log10, yscale = :log10,
        xlabel="σα²",
        ylabel="σμ²",
        zlabel="log₁₀(LL)",
        # zlims = (-3.4849, -3.48488),
        title="Log Likelihood")
    p3
    savefig(p3, "../../output/estimation_notes/2022-09-13 3d LL plot of starting values 1.pdf")

    # p2 = plot(contour(σₐ²vec, σᵤ²vec, Z, fill = (true,cgrad(:Spectral, scale = :log))),
    #     xlabel="σₐ² starting value",
    #     ylabel="σᵤ² starting value",
    #     title="Log Likelihood values based on starting param values");
    # p2

    p1 = contour(σₐ²vec, σᵤ²vec, LL10_plot, fill = true)
    # p3 = contour(x, y, (x,y)->cos(x)+cos(y), fill=(true,cgrad(:Spectral, scale = :log)))
    filepath = "../../output/estimation_notes/2022-09-13 3d LL plot of starting values.pdf"
    p = plot(p1,
        xlabel="σₐ² starting value",
        ylabel="σᵤ² starting value",
        title="Log Likelihood values based on starting param values");
    savefig(p, filepath)
end
# plot_ll_big_sigmas()


function trend_vec(β)
    boi = first(β, 4)
    β = last(β, 2)
end

function plot_real_data()
    N=4; T=60
    data = read_data(N, T)
    p = @df data plot(:t, :eᵢₜ, group=:i,
        xlabel="year",
        ylabel="emisssions",
        title="Observed Regional Emissions")
        
    V = Diagonal(ones(length(data.eᵢₜ)))
    linear_formula = @formula(eᵢₜ ~ 1 + i + t + t^2)
    gls = mygls(linear_formula, data, V, N)
    v = vec(gls[:resid])
    data.v_est = v
    p2 = @df data plot(:t, :v_est, group=:i,
        xlabel="year",
        ylabel="emisssions",
        title="De-trended Observed Regional Emissions")
    
    # plot trend lines on top of data
    data.e_detrend = gls[:yhat]
    @df data plot!(p, :t, :e_detrend, group=:i)
end

function plot_simulated_data()
    ρ = 0.8785
    σₐ² = 10^10
    σᵤ² = σₐ²
    b₀ = [378017.714, -45391.353, 347855.764, 314643.98]
    β = [30_000., -6.6]
    N=4; T=60

    # Generate simulated data from realistic parameter estimates
    data = dgp(ρ, σₐ², σᵤ², β, N, T; v₀ = 10.0^6, b₀ = b₀)
    p = @df data plot(:t, :eᵢₜ, group=:i,
        xlabel="year",
        ylabel="emisssions",
        title="Simulated Regional Emissions")

    # Demean and detrend
    V = Diagonal(ones(length(data.eᵢₜ)))
    linear_formula = @formula(eᵢₜ ~ 1 + i + t + t^2)
    gls = mygls(linear_formula, data, V, N)
    v = vec(gls[:resid])
    data.v_est = v
    p2 = @df data plot(:t, :v_est, group=:i,
        xlabel="year",
        ylabel="emisssions",
        title="De-trended Simulated Regional Emissions")
    
    # plot trend lines on top of data
    data.e_detrend = gls[:yhat]
    @df data plot!(p, :t, :e_detrend, group=:i)
end


function plot_gradient(v)
    ρ = 0.8785
    σₐ²vec = ℯ.^(-8:1)
    σᵤ² = 3.6288
    N=4; T=60
    # v = 2
    temp(σₐ²) = ∂nLL∂y("σₐ²", ρ, σₐ², σᵤ², v, N, T)
    grad = temp.(σₐ²vec)
    p = plot(σₐ²vec, grad,
        xlabel="σₐ²",
        ylabel="gradient",
        title="Gradient Values over σₐ² (σᵤ² = $σᵤ²)")
    p

    # Show gradient surface
    σᵤ²vec = ℯ.^(1:0.5:3)
    temp2(σₐ², σᵤ²) = ∂nLL∂y("σₐ²", ρ, σₐ², σᵤ², v, N, T)
    X = repeat(reshape(σₐ²vec, 1, :), length(σᵤ²vec), 1)
    Y = repeat(σᵤ²vec, 1, length(σₐ²vec))
    Z = @showprogress map(temp2, X, Y)
    p2 = surface(σₐ²vec, σᵤ²vec, Z, xscale = :log10, yscale = :log10,
        xlabel="σα²",
        ylabel="σμ²",
        zlabel="σα² gradient",
        title="Negative Log Likelihood Gradient")

    x = 1:4
    y = 5:10
    X = repeat(reshape(x,1,:), length(y), 1)
    Y = repeat(y, 1, length(x))
    Z = map((x,y) -> x^2 + y^2, X, Y)
    surface(x,y, Z)

end