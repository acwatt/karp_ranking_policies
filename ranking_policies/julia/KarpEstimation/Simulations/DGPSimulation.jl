include("../Model/Model.jl")



function check_fixed_effects()
    # Define paramters with zero error variance to get smooth data
    N, T = 4, 60
    b₀ = [3.146, -0.454, 3.78, 3.479]  # from test_starting_real_estimation() for n=4, after changing emissions units to 10,000 tons
    β = [0.265, 0] # from test_starting_real_estimation() for n=4
    ρ = 0.8785                        # from r-scripts/reproducting_andy.R  -> line 70 result2
    σₐ² = 0  # from test_starting_real_estimation()
    σᵤ² = 0  # from test_starting_real_estimation()
    @info "check_fixed_effects()"     b₀ β ρ σₐ² σᵤ²

    # Generate simulated data
    simulated_data = dgp(ρ, σₐ², σᵤ², β, N, T; v₀ = 0.0, b₀ = b₀)

    # Estimate fixed effects
    plist = [ρ, σₐ², σᵤ²]
    lower_bounds = [0.878, 1e-4, 1e-4]
    linear_formula = @formula(eᵢₜ ~ 1 + i + t + t^2)
    V = Diagonal(ones(length(simulated_data.eᵢₜ)))
    gls = mygls(linear_formula, simulated_data, V, N)
    β2, b₀ᵢ = translate_gls(gls, N)

    # Predict values using estimated parameters
    simulated_data2 = dgp(ρ, σₐ², σᵤ², β2, N, T; v₀ = 0.0, b₀ = b₀ᵢ)

    # Compare differences
    maximum(abs.(simulated_data.eᵢₜ - simulated_data2.eᵢₜ))

    # Compare plots
    @df simulated_data plot(:t, :eᵢₜ, group = :i)
    @df simulated_data2 plot!(:t, :eᵢₜ, group = :i)
end