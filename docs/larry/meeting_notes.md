# 2022-08-01 Zoom Meeting
present: A. Watt, L. Karp



## Next:
- simulated data but not too small of a dataset
- plot the LL function for n=2, T=5, rho=0.878 along sigmas

Bruce Hanson Econometrics (for time series and MLE)


___
# 2022-01-20 Zoom Meeting
present: A. Watt, L. Karp

## Notes on covariance matrix equations
Two quick notes on what I think are typos in the most recent version of notes (sent in December):
1. In the last line of equation 9 of section 1.3.4 (in the derivative w.r.t. sigma_u), it looks like (1-kappa)*iota should be outside the parentheses and not multiplied by ρ^s / n. 

2. Equation (7) on page 5 shows the maximization of the log likelihood, both terms are of the same sign. The matrix derivative identity used on the very bottom of page 8 flips the sign of the middle term in the derivative of the log likelihood. The matrix derivative identity used at the top of page 9 does not flip the sign of the third term. Therefore I think the signs of the terms in the last equation on page 9 (the derivative of the log likelihood) should be different. I believe the first term should be negative, and the second should be positive.

I've proceeded assuming these two items, but correct me if I'm mistaken. It would be easy to correct the code.

## Results
1. Estimating the parameters on the real data using the iterative method we've discussed resulted in estimates that depended somewhat on the starting value of the search:
    - rho in the neighborhood of 0.12 to 0.99
    - sigma_a in the neighborhood of 20 to 168
    - simga_u in the neighborhood of 200 to 290

2. Implementing gradient descent had no appreciable effect on estimation: among an array of different estimations with simulated data, the non-gradient (Nelder-Mead) minimization resulted in the same parameter estimates as the numerical gradient descent and analytical gradient descent methods (all three returned basically the same values, down to about the 6th or 7th decimal place). 

    So this still leaves us with estimates for sigma_a that aren't very accurate at N=2, T=50 (e.g., when rho=0.85 and simga_a = simga_y = 1, average estimates are around rho=0.425, simga_a=0.655 and sigma_u=1.017, respectively). However, I do feel comfortable that I correctly coded the analytical gradient of the log likelihood given that the analytical gradient descent aligns so well with both the numerical gradient descent and the non-gradient descent methods.

3. I am in the process of testing how close the estimates are when realistic "true" parameter values are used to generate the data, estimates taken from the data. Here are the realistic parameter estimates from a first run pass at estimating the parameters:
    - b₀ = [200000, 560000]  # Estimated from GLS-MLE iterations (US and EU fixed effects)
    - β  = [22000, -100]     # Estimated from GLS-MLE iterations (linear and quadratic time trend)
    - ρ  = 0.8785            # Estimated from aggregate data, same method as Andy
    - σₐ² = 40               # Estimated from GLS-MLE iterations, using various starting values
    - σᵤ² = 265              # Estimated from GLS-MLE iterations, using various starting values

    These values are used to generate simulated data. I am now running some estimations on that simulated data with different starting search values for σₐ² and σᵤ². I'm not sure if these results will be done by our meeting -- it's taking much longer on the simulated data for reasons I haven't been able to determine. 


In any case, it's exciting that the basic estimation results are not zero! But none of these estimates have standard errors yet and I'm still hoping to show (at least numerically) that this can be a consistent estimator.

## Next
- save first-past ols GLS estimates, and final GLS estimates (DONE)
- Use N=4, real data, hold rho constant at 0.8785, estimate other params
    - with and without gradient 
    - different starting search values
    - round to 3 decimals (DONE)
- plot log likelihood as a function of sigma_a, sigma_u, holding rho constant
- Think about delta method vs taylor series expansion (want to know about momemnts of f(sigma_a, sigma_u))





___
# 2022-01-04 Zoom Meeting
present: A. Watt, L. Karp

## Progress
- have completed an example estimation using the two-step procedure outlined in [Larry's recent note](https://github.com/acwatt/karp_ranking_policies/blob/main/docs/larry/2021-12_new_writeup_after_statistics_consulting.pdf).
- Have hard-coded a function that generates the covariance matrix given rho and sigma parameters for N=2, T=3
- Uses simulated data to test estimation of rho and sigmas

# Discussion
- we can add a rho estimation step before the sigma ML estimation, using aggregate data. Then only need to estimate sigmas
- Need to try with many more time periods because T=3 might be showing underidentification issues.
    - Need to create more generic covariance matrix function for arbitrary N,T using the general form from notes.
- Try estimating b_0i and beta time trends from the data -- at least to get a sence of magnitude.
    - Pick two b_0i to use for N=2 and use estimated beta time trends.


## Next steps
1. Generalize the variance matrix function to arbitrary N and T, check N=2, T=3 case to see if it's consistent with Larry's note.
2. Test N=2, T=50.
3. Pick b_0i fixed effects and beta time trends consistent with data (or just set equal to 0)
4. Set rho = 0.85, and MLE sigmas
5. Use gradient decent to see if it converges to non-zero sigmas
6. Go back and add rho estimation with aggregate data before iterations (since it won't change?)
    - maybe rho would change if we were using GLS with updated covariance matrix...


## Questions
- Can you preperly decouple the AR(1) trend parameter rho from the time trend parameters?
    - "I've seen this done before, but why is it legitimate to detrend and demean the data before estimating the rest of the model?"
- Can we jointly estimate the b_0i's, betas, rho, and sigmas?
