Oberhofer and Kmenta, “A General Procedure for Obtaining Maximum Likelihood Estimates in Generalized Regression Models.”
- Mentioned by Greene (7th ed) in 14.9.2 Generalized Regression Model, referring to how to estimate the GRM using MLE. Greene says about this paper, "... who showed that under some fairly weak requirements, *most importantly that \theta not involve \sigma^2 or any of the parameters in \beta*, this procedure would produce the [MLE]".
- issue: \theta = (rho, lambda) in \Omega(\theta) involves lambda, which is \sigma_a^2/b^2. But \sigma^2 was defined as \sigma_a^2/b^2, so \theta includes lambda, which inlcuses simga_a^2, which is in \simga^2. With the new notes from Larry, he doesn't factor out the simga_a. Need to better understand the estimation of \hat{\sigma^2} in Greene (ed 7, pg553) and in this paper to know if the estimated sigma^2 is correlated with sigma_a in V from Larry's notes.


ec1-11.pdf: lecture notes on estimation procedures.
See pg 20 for the Newey-West beta SE estimator.
