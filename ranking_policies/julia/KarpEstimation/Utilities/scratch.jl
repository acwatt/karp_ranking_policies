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
