## ---------------------------
## R version 3.6.3 (2020-02-29)
## Script name: reproducing_andy.R
##
## Purpose of script: reproduce Andy Hultgren's empirical work for this project
##
## Author: Aaron Watt
## Email: aaron@acwatt.net
##
## Date Created: 2021-07-30
##
## ---------------------------
## Notes:
##


## PACKAGES ===========================================
# install.packages("pacman")
# install.packages("car", repos="http://R-Forge.R-project.org")
library(tidyverse)
library(dplyr)
library(car)
library(ggplot2)
library(reshape2)
library(fastDummies)
library(fs)


## WORKING DIRECTORIES ===============================
# use working path to find root project folder
path_parts = path_split(path_wd())[[1]]
root_index = match(c("karp_ranking_policies"), path_parts)
root_path = path_join(path_parts[1:root_index])

data71_path = path(root_path, 'data', 'andy', 'clean', 'ts_allYears_nation.1751_2014.csv')


## PREPARE DATA =====================================
min_year = 1945
max_year = 2005

data1 = read.csv(data71_path)
data1 = data1 %>% filter(Year >= min_year & Year <= max_year)
plot(data1$Year, data1$ebar, type="n", 
     main="Average Emissions (ebar, actual data)", 
     xlab="Year", ylab="tons of C02")
lines(data1$Year, data1$ebar)


## EQ 71 ESTIMATIONS ======================================
ebar = data1$ebar
ebar_lag = data1$ebar_lag
t = data1$time
t_lag = data1$time_lag

model1 = ebar ~ rho*ebar_lag + alpha + d1*t + d2*t^2 + rho*d1*t_lag + rho*d2*t_lag^2
model2 = ebar ~ rho*ebar_lag + alpha + d1*t + d2*t^2 - rho*d1*t_lag - rho*d2*t_lag^2

starting_values = list(rho=0.878, alpha=-650, d1=10.75, d2=-0.0275)
# starting_values = list(rho=0.1, alpha=1, d1=1, d2=1)
lower_bounds = list(0, -1000, 0, -5)
upper_bounds = list(1, 1000, 1000, 5)

result1 = nls(model1, start=starting_values, lower=lower_bounds, upper=upper_bounds,
              algorithm="port", control=list(maxiter=1000))
result1

result2 = nls(model2, start=starting_values, lower=lower_bounds, upper=upper_bounds,
              algorithm="port", control=list(maxiter=1000))
result2

B0_1 = car::deltaMethod(result1, "alpha/(1-rho)")[1,1]
se_B0_1 = car::deltaMethod(result1, "alpha/(1-rho)")[1,2]
B0_2 = car::deltaMethod(result2, "alpha/(1-rho)")[1,1]
se_B0_2 = car::deltaMethod(result2, "alpha/(1-rho)")[1,2]

# Calculate sigma^2 / b^2 in the paper (note the B1^2 will drop out when calculating the ratio k).
resids1 = predict(result1) - ebar
resids2 = predict(result2) - ebar

sigma1_sq = sum(resids1^2) / (max_year - min_year - 3 - 2)
print(paste("Estimate of sigma^2 / B1^2 for ending year", max_year, ":", sigma1_sq ))
sigma2_sq = sum(resids2^2) / (max_year - min_year - 3 - 2)
print(paste("Estimate of sigma^2 / B1^2 for ending year (corrected sign)", max_year, ":", sigma2_sq ))

# Estimate the standard error of sigma^2 / b^2, following Eq (3) in https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
se_sigma1_sq = (sigma1_sq / 2*(max_year - min_year - 3 - 2))^0.5
se_sigma2_sq = (sigma2_sq / 2*(max_year - min_year - 3 - 2))^0.5

# ebar_hat = predict(result1)
# df = data.frame(t, ebar, ebar_hat)
# df <- melt(df ,  id.vars = 't', variable.name = 'series')
# ggplot(df, aes(t,value)) + geom_line(aes(colour = series))
# 
# plot(t, ebar)
# lines(t, predict(result1), col=2)
# 
# plot(t, resids1)



## EQ 77 ESTIMATIONS ======================================
data77_path = path(root_path, 'data', 'andy', 'clean', 'grouped_allYears_nation.1751_2014.csv')
data2 = read.csv(data77_path)
data2 = data2 %>% filter(Year >= min_year & Year <= max_year)
# data2 <- dummy_cols(data2, select_columns = 'group')

# Calculate average emissions in each year, and subtract from each region's emissions.
data2 = data2 %>% 
  group_by(group) %>% 
  mutate(co2_demeaned = CO2 - mean(CO2))
data2 <- data2[order(data2$Year, data2$group),]
data2 = subset(data2, group != "USA")


# NEXT:
#  - reproduce GLS estimation (easier way than andy's 137+ code?)

mod.ols <- lm(ebar ~ t + t^2)
summary(mod.ols)
plot(t, residuals(mod.ols), type="o", pch=16,
     xlab="Year", ylab="OLS Residuals")
abline(h=0, lty=2)

acf(residuals(mod.ols))
acf(residuals(mod.ols), type="partial")
durbinWatsonTest(mod.ols, max.lag=20)














