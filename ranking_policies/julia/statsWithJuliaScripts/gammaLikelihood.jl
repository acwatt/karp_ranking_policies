using Random, Distributions, Gadfly, LaTeXStrings; pyplot()
Random.seed!(0)

actualAlpha, actualLambda = 2,3
gammaDist = Gamma(actualAlpha,1/actualLambda)
n = 10^2
sample = rand(gammaDist, n)

alphaGrid = 1:0.02:4
lambdaGrid = 2:0.02:6

likelihood = [prod([pdf.(Gamma(a,1/l),v) for v in sample])
                        for l in lambdaGrid, a in alphaGrid]
                        
p = surface(alphaGrid, lambdaGrid, likelihood, lw=0.1, 
	c=cgrad([:blue, :red]), legend=:none, camera = (135,20),
	xlabel=L"\alpha", ylabel=L"\lambda", zlabel="Likelihood");

@gif for i in range(0, stop=360, length=1000)
    surface!(p, camera = (i, 20))
end every 10

@gif for i in range(0, stop = 2π, length = n)
    f(x, y) = sin(x + 10sin(i)) + cos(y)

    # create a plot with 3 subplots and a custom layout
    l = @layout [a{0.7w} b; c{0.2h}]
    p = plot(x, y, f, st = [:surface, :contourf], layout = l)

    # induce a slight oscillating camera angle sweep, in degrees (azimuth, altitude)
    plot!(p[1], camera = (10 * (1 + cos(i)), 40))

    # add a tracking line
    fixed_x = zeros(40)
    z = map(f, fixed_x, y)
    plot!(p[1], fixed_x, y, z, line = (:black, 5, 0.2))
    vline!(p[2], [0], line = (:black, 5))

    # add to and show the tracked values over time
    global zs = vcat(zs, z')
    plot!(p[3], zs, alpha = 0.2, palette = cgrad(:blues).colors)
end