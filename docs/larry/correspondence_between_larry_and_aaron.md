# 2022-10
> On Wed, Oct 12, 2022, 8:49 PM Larry S. KARP <karp@berkeley.edu> wrote:

Hi Aaron,

I've been thinking more about our conversation.  It's late and this has been a long day, so I hope that this note is coherent.

My paper is clear that there is a particular type of indeterminacy.  See marked passage on page 10 in the paper that I will send you after I finish this email.  This indeterminacy does not mean that the model is unidentified -- even in the static setting.  For example, if I set sigma^2_{alpha} = 0 and sigma^_{mu}/n = sigma^2, there will be a lot of variation in emissions over regions. If I set  sigma^2_{alpha} = sigma^2 and sigma^_{mu}/n = 0 there will be no cross-regional variation in emissions.  However, the variation in aggregate emission will be the same in these two models.  (See eqn 7.)

My point is that it should be possible to identify the pair of variances that satisfy eqn 7 and best describe the data.  One of the experiments that we discussed should provide an example of the fact that the time series looks quite different, depending on which combination of parameters we choose (that satisfy eqn 7).

> Aaron response: This is because the set identification is non-linear -- it does not mean either $\sigma_u$ or $\sigma_a$ can be zero. The set is identified where $\sigma^2_{a,1} = 0, \sigma^2_{u,1}>0$ and $\sigma^2_{a,2} > 0, \sigma^2_{u,2}>\sigma^2_{u,1}>0$ (the same cross-regional variation in the data can be explained by increasing $\sigma^2_{a}$ and decreasing $\sigma^2_{u}$). But you cannot make $\sigma_u=0$ and still explain the cross-regional variation. The issue is that $\sigma_u$ can explain both cross-regional and cross-year variation. But $\sigma_a$ can only explain cross-year variation.

So I am still unable to explain your (and Sunny's) results for MLE.  However, I recall that Andy used a particular approach in the two-part estimation. He estimated eqn 75 (aggregate data) using GLS (not by "transforming the data" as described on page 22 of the appendix C5).  That's good -- GLS is more efficient than the "transforming the data" approach.  This procedure produces an estimate of sigma^2 = sigma^2_{alpha} + sigma^2_{mu}/n, rho, B_0, and h(t).

However, I am not sure how he estimated the parameter sigma^2_mu in the system 81 (page 26 of the appendix).  I know that he used the cholesky decomposition -- because I provided him with the formula for the matrix V.  It is somewhere in the notes, but should be easy to produce, using the formula for OMEGA, see page 25 of the appendix.  I think that he used a numerical routine with GLS.  That SHOULD work, but who knows with numerical routines...  I am pretty sure that he never got around to using the closed form expression for the estimator of sigma^_{mu} that I provided in equation 77 on page 24 of the text.  Using that closed form expression requires a bit of programming (not using a canned package).  But I am sure that the programming is quite simple.  I had always intended to ask someone to use this closed form expression (rather than a canned program) to estimate sigma^2_mu, but I forgot about that request (when I decided that MLE offered a better chance).

In summary it would be great if you would (i) re-estimate eqn 75 using GLS  to obtain an estimate of sigma^2  and use eq 77 to estimate sigma^_{mu}.  (You would also need to use 76 and the estimates of B_0 and h(t) to recover the region-specific fixed effects.)  I'll keep my fingers crossed.  

> see 2022-12-21 meeting notes.docx


> On Wed, Oct 12, 2022 at 2:14 PM Aaron Watt <aaron.watt@berkeley.edu> wrote:

Hello Larry!

Here's a quick update in preparation for our meeting today:
- I was able to test several different optimization algorithms and all converged on the same likelihood-maximizing point: sigma_a^2 = 0, sigma_u^2 ~ (180,000 tons CO2)^2
- I believe the joint error term's variance is identified in this model, which means sigma_a and sigma_u would be set identified.
- In the past, we've tried estimating sigma_a with aggregate data, and get sigma_a larger than the joint error variance estimated from this model (sigma_a^2 + sigma_u^2/n). This implies that sigma_u is then zero.
- I think we can combine both of these estimations in either MLE or in GMM. But it appears to me that the GMM implementation is simpler than writing a new log likelihood function for the joint estimation of both the regional estimation and the aggregate estimation.
- So I am going to suggest in the meeting that we might try a GMM estimation to jointly estimate sigma_a and sigma (the joint error variance).
