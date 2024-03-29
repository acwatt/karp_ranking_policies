## ---------------------------
## R version 3.6.3 (2020-02-29)
## Script name: simulations.R
##
## Purpose of script: simulate emissions data and estimate to recover paramters
##
## Author: Aaron Watt
## Email: aaron@acwatt.net
##
## Date Created: 2021-08-30
##
## ---------------------------
## Notes:
##

library(dplyr)
library(fs)
library(emdbook)

path_parts = path_split(path_wd())[[1]]
root_index = match(c("karp_ranking_policies"), path_parts)
root_path = path_join(path_parts[1:root_index])
output_path = path(root_path, 'output')
setwd(path(output_path, 'simulation_plots'))


set.seed(198911)


N = 4  # Number of regions
start_year = 1751
min_year = 1945
max_year = 2005
end_year = 2014

T = end_year - start_year + 1  # Number of time periods
Tvec = start_year:end_year - 1900


# FOR LOOP TO EXPLORE HOW INITIAL PARAMTERS EFFECT THE ESTIMATION ACCURACY
df_results = data.frame()
    
for (sigma_a in c(0.1, 1, 5, 10, 20, 50, 100, 500)) {
  for (sigma_u in c(0.1, 1, 5, 10, 20, 50, 100, 500)) {
    # PARAMETERS
    b = 1  # scaling param
    rho = 0.9
    d1 = 5
    d2 = 2
    beta = c(d1, d2)
    nu0 = 100
    b0mean = 20
    b0sd = 10
    b0 = rnorm(n=N, mean=b0mean, sd=b0sd)  # b_0i
    df_temp = data.frame(sigma_a=sigma_a, sigma_u=sigma_u, nu0=nu0, b0mean=b0mean, b0sd=b0sd)
    
    # Fill region characteristics (just time, time^2 at this point)
    X = array(dim=c(N, T, 2))
    for (i in 1:N) {
      for (t in 1:T) {
        X[i,t,] = c(Tvec[t], Tvec[t]^2)
      }
    }
    
    # alpha_t
    a = rnorm(n=T, mean=0, sd=sigma_a)
    # mu_i,t
    mu = matrix(rnorm(n=T*N, mean=0, sd=sigma_u), nrow=N)  # rows are regions, cols are times
    # theta_t
    theta = colSums(mu)/N
    # Build nu_t
    nu = c(rho*nu0 + a[1] + theta[1])
    for (t in 2:T) { 
      nu[t] = rho*nu[t-1] + a[t] + theta[t]
    }
    # Build nu_it
    nu_it = array(dim=c(N, T))
    for (i in 1:N) {nu_it[i,1] = rho*nu0 + a[1] + mu[i,1]}
    for (i in 1:N) {
      for (t in 2:T) {
        nu_it[i,t] = rho*nu[t-1] + a[t] + mu[i,t]
      }
    }
    
    # BUILD USING DATAFRAME (COME BACK TO COMPLETE)
    df = data.frame(Year = start_year:end_year, time = start_year:end_year - 1900)
    df$time2 = df$time^2
    df = df %>%
      mutate(time_lag = lag(time), time2_lag = lag(time2))
    df$alpha = rnorm(n=T, mean=0, sd=sigma_a)
    for (i in 1:N) {
      df[paste0("mu_", i)] = rnorm(n=T, mean=0, sd=sigma_u)
    }
    
    # EMISSIONS DATA
    e = array(dim=c(N, T))
    for (i in 1:N) {
      for (t in 1:T) {
        e[i,t] = (b0[i] +  beta%*%X[i,t,]) / b + nu_it[i,t] / b
      }
    }
    
    B0 = sum(b0) / N
    alpha = (1-rho) * B0
    
    ebar = colSums(e) / N
    ebar_lag = append(NA, ebar[1:(T-1)])
    df = data.frame(ebar, time=1:T, Year = start_year:end_year)
    df = df %>%
      mutate(ebar_lag = lag(ebar), time_lag = lag(time))
    
    
    plot_name = paste('rho', rho,
                      'sigma_a', sigma_a, 'sigma_u', sigma_u,
                      'b0mean', b0mean, 'b0sd', b0sd,
                      'nu0', nu0,
                      'beta', beta[1], beta[2])
    jpeg(paste0(plot_name, '.jpg'), width = 1380, height = 700)
    plot(df$Year, df$ebar, type="n", 
         main = "Average Emissions (ebar, simulated data)", 
         xlab = "Year", ylab = "tons of C02",
         sub = plot_name)
    lines(df$Year, df$ebar)
    dev.off()
    
    df = df %>% filter(Year >= min_year & Year <= max_year)
    jpeg(paste0(plot_name, '_used.jpg'), width = 1380, height = 700)
    plot(df$Year, df$ebar, type="n", 
         main = "Average Emissions (ebar, simulated data)", 
         xlab = "Year", ylab = "tons of C02",
         sub = plot_name)
    lines(df$Year, df$ebar)
    dev.off()
    
    
    # ESTIMATION ================================================
    
    
    # find good starting values
    model_start = ebar ~ ebar_lag + time + I(time^2) + time_lag + I(time_lag^2)
    result_start = lm(model_start, data = df)
    print(summary(result_start))
    
    model = ebar ~ rho*ebar_lag + alpha + d1*time + d2*time^2 - rho*d1*time_lag - rho*d2*time_lag^2
    coef = result_start$coefficients
    starting_values = list(rho = rho, #coef['ebar_lag']
                           alpha = alpha, #coef['(Intercept)']
                           d1 = coef['time'], 
                           d2 = coef['I(time^2)'])
    result = nls(model, data = df, start=starting_values)
    summary(result)
    print(paste('True rho, alpha, d1, d2:', rho, alpha, beta[1], beta[2]))
    
    plot(df$Year, df$ebar)
    lines(df$Year, predict(result))
    lines(df$Year, predict(result_start))
    
    B0_est = car::deltaMethod(result1, "alpha/(1-rho)")[1,1]
    B0_est
    B0_se = car::deltaMethod(result1, "alpha/(1-rho)")[1,2]
    B0
    
    # Add results to dataframe
    for (param in c('rho', 'alpha', 'd1', 'd2')) {
      true = eval(parse(text = param))
      est = coef(summary(result))[param, 'Estimate']
      se = coef(summary(result))[param, 'Std. Error']
      diff = abs(true - est)
      df_temp[param] = true
      df_temp[paste0(param, '_est')] = est
      df_temp[paste0(param, '_se')] = se
      df_temp[paste0(param, '_pval')] = coef(summary(result))[param, 'Pr(>|t|)']
      df_temp[paste0(param, '_diff')] = diff
      if (diff < 1.96*se) {
        in_CI = 1
      } else {
        in_CI = 0
      }
      df_temp[paste0(param, '_inCI')] = in_CI
    }
    df_temp$B0 = B0
    df_temp$B0_est = B0_est
    df_temp$B0_se = B0_se
    df_temp$B0_diff = abs(B0 - B0_est)
    if (df_temp$B0_diff < 1.96*B0_se) {
      df_temp$B0_inCI = 1
    } else {
      df_temp$B0_inCI = 0
    }
    df_results = rbind(df_results, df_temp)
    
  }  # END FOR LOOPS
}

# Save results to file
write.csv(df_results, "simulation_results.csv", row.names = FALSE)





























