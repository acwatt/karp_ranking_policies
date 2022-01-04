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
