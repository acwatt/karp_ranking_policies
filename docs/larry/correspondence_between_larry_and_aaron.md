# 2023-01-16
On Thu, Dec 22, 2022 at 10:39 PM Larry S. KARP <karp@berkeley.edu> wrote:

Hello Aaron,

Thanks for all of this work!  Can I ask you to do the following (roughly in order of importance)

1) We had discussed an experiment that you might have done early on in this process ... . Use simulated data where sigma_{alpha} = sigma_{mu} and see if either of our estimation methods (ML and the "disjoint" method) can recover the correct parameters.  For this experiment I think that the region specific effects can be ignored. If either estimation method can recover reasonable estimates of the sigmas and the rho, then we can conclude that the empirical results you have generated are telling us something about the world: sigma_a really is approximately zero.  But if neither estimation method can recover the parameters, then I guess that we can't conclude anything from the data.  (But I sure would like to know why the estimation fails!)

> Aaron: I did some simulations of the rho estimation that I didn't include in the report. I'll add those to the next report -- the general finding being that the estimate of rho is biased downward for most simulated values of rho, but the true simulated rho value is almost always in the 95% confidence interval. The estimate of rho has a strong interplay with the time trend parameters. My main concern here is the ripple effect of the rho, time trend, and regional fixed effects estimations. If those are biased, I generate incorrect regression errors. The errors are the only data used to estimate the sigmas and generate the likelihood function. So my recent focus has been to try to ensure both the rho and regional fixed effects estimations are as least biased as possible. I've been frustrated by what seems like a simple estimation generating biased results for the rho, time trend, and regional fixed effects. I've been looking through the literature and it seems that the biasedness of autocorrelation coefficient estimators is known and I haven't found a good source on correcting that bias (please let me know if you have a good source and I'll keep looking). I'll clean up my simulations and send you results to document the bias that I see.
> > Larry: I was not aware of this problem, so I don't have any suggestions for a fix. I see that biases estimates produce biased residuals, but I have no idea why that would create the strange results we have seen.

>I'll then work on doing a write up of the sigma simulation estimations for both the ML and disjoint methods, where sigma_alpha = sigma_mu. I'll ignore regional fixed effects and time trend and just generate simulated data based on generated nu's (from generated alpha's and mu's).
>> Larry: Great, I am really interested in seeing a systematic treatment of simulations.  I think that we have seen some examples earlier, but not enough detail to tell a story.  It would be great to understand what is going on.  But if the simulations show that the estimation methods do a reasonable job, then we can conclude that the finding that sigma_a = 0 is a real feature of the data.  And if the simulations show that the estimation methods produce sigma_a = 0 even when the true value is positive, then we know that there is a problem with the estimation methods, and we cannot trust them.  Ideally, we could find an estimation method that does  work, but that may be too much to ask.  At this point I would be (more or less) satisfied with either conclusion: (i) the methods work, and the true sigma_a = 0; or (ii) the methods do not work, so we really cannot say anything.

2) Thanks for the write-up. Could you prepare a "replication appendix" for the two estimation methods, a folder that (down the road) I can include with a submission.  Please be painstaking with this.  This would include a latex document explaining everything (with equation references) and a tediously documented code.  I will eventually revise the latex appendix, but I will  not be able to check the code in any useful way. However, it is a real possibility that a referee would look at it and attempt to replicate the results.  This might be years from now, when you have long since moved on. It would be a big drag if at that time things got held up because the referee could not implement the code. 

>Aaron: I'll start working on the replication appendix! I'll reference equation numbers from the October 13, 2022 draft of the paper.

3) I'm glad that you mentioned the MOM. About 18 months ago I prepared a document (attached) to explain the estimation problem to a "statistical consultant" (a grad student in stat); but I'm afraid that he was not much use.  The first method that I suggested was a MOM estimator.  I used two possible MOM estimators in sections 3.1 and 3.2. (These are slightly different because in the first alternative I divided by an estimate of sigma^2 and in the second I did not.)  It seems straightforward to write down the moment condition -- see eqn 4 in the attachment, and the unnumbered equation on the next page for the second alternative.  However, I do not know the theory, so what seems reasonable to me might  be simply wrong. Also, because the moment condition is a matrix equation, it was not obvious what maximand to use.  So I picked the Frobenious norm.  

It would be great if you would implement one or both of the MOM estimators (and also check them using simulated data as in item 1).

>Aaron: After points 1 and 2, I'll start working on the MOM estimators and test using some simulations.
>>Larry: That would be great. 






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
