# 2022-12-15
- need to replicate Andy's code
- Tested simulated data after adjusting the scale of the data (divided by 10^5)
  - seem to be converging on positive values of sigmas -- Should plot the gradient, seems to go to zero at σₐ² = 2.2568065 and σᵤ² = 3.3751489 -- does that hold for lower values of σₐ²?
    - The gradient for σₐ² seems totally dependent on the value for σᵤ² (grad σₐ² high at low values of σᵤ²). Should definitely plot
    - Try different optim algos: currently using BFGS, try all others at a few combinations of starting sigmas that go in wierd directions
  - should test starting value sensitivity of both real and simulated
  - should test on real data
    - seems very sensitive to starting sigma values. Should plot the gradient
    - 

Coding:
- should setup debug log and change inner functions from @info to @debug


Autocorrelation:
- see http://web.vu.lt/mif/a.buteikis/wp-content/uploads/2019/11/MultivariableRegression_4.pdf pg 77
- plot estimated rhos on global avg data
- looks like there's at least 2 lags

Gradient for real data:
```
Starting MLE with 0.8785, 0.01, 0.01
Analytical gradient: [-7734.590666270136, -4.3041855681906575e6]

Starting MLE with 0.8785, 7536.029406194092, 4.303987006912751e6
Analytical gradient: [1.6546691603988224e-5, 2.785212926555983e-5]
```

NEXT:
- see 2022-12-21 meeting notes.docx and end of serial_correlation_tests.jl for most recent work.
- use ρ est and lb and ub from serial_correlation_tests to est σ's in MLE
- Can I encorporate serial_correlation_tests MB ρ procedure in MLE?


```