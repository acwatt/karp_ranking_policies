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


## PACKAGES ====================================================================
#install.packages("dummies")
library(dummies)


## PREPARE DATA ================================================================
data <- read.csv("../../data/sunny/clean/grouped_nation.1751_2014.csv")

# Take only years in [1945, 2005]
data<-subset(data, Year>= 1945 & Year<=2005)

# Reset time variable to 1 at beginning of subset
data$time<- data$time-44

# Reorder the data to be in time order
data <- data[order(data$time),]

# Create indicators for each of the 4 regions
data$gp <- factor(data$group)
as.numeric(data$gp)
b <- dummy(data$gp, sep = "_")  # b_0i (not b, b set to 1 for estimation)

#x1 <- matrix(data = data$gp_BRIC, ncol = 1)
#x2 <- matrix(data = data$gp_EU, ncol = 1)
#x3 <- matrix(data = data$gp_Other, ncol = 1)
#x4 <- matrix(data = data$gp_USA, ncol = 1)

# Combine region indicators with t and t^2 into matrix
t <- matrix(data=data$time, ncol=1)
t2 <- t^2
X <- cbind(b,t,t2)

# Make vector of emissions (our y variable)
e <- matrix(data=data$CO2, ncol=1)


## ESTIMATION ==================================================================
# Setings for estimation
n <- 4  # number of regions
T <- 60  # number of time periods (years)
nt <- n*T
rep <- 40 # max number of iterations to get to final estimate

# Create empty matrices to be filled in during estimation iterations
V <- matrix(1:(nt*nt), nrow = nt)  # . . . . . . . . . . Covariance matrix (V)
dVdrho <- matrix(1:(nt*nt), nrow = nt)  #  . . . . . . . rho derivative of V
dVdlambda <- matrix(1:(nt*nt), nrow = nt)  # . . . . . . lambda derivative of V
a <- matrix(1:(2*rep), nrow=rep)  #  . . . . . . . . . . a1 and a2, time trend params (or grid over rho and lambda to search?)
rho <- matrix(1:rep, nrow = rep)  #  . . . . . . . . . . AR1 process parameter
lambda <- matrix(1:rep, nrow = rep)  # . . . . . . . . . sigma_u^2 / sigma_a^2
sigma2 <- matrix(1:rep, nrow = rep)  # . . . . . . . . . sigma^2 = sigma_a^2 / b^2 (not b_0i, b set to 1 for estimation)
lnL_first_term <- matrix(1:rep, nrow = rep)  # . . . . .  
lnL_second_term_second <- matrix(1:rep, nrow = rep)  # . 
lnL_second_term <- matrix(1:rep, nrow = rep)  #  . . . . 
lnL_third_term <- matrix(1:rep, nrow = rep)  # . . . . . 
lnL <- matrix(1:rep, nrow = rep)  #  . . . . . . . . . . 
grad <- matrix(1:(2*rep), nrow = rep)  # . . . . . . . . gradient (never used?)
step <- matrix(1:rep, nrow = rep)  # . . . . . . . . . . step (never used?)
accuracy <- matrix(1:rep, nrow = rep)  # . . . . . . . . 

a[1,1] <- 0.1
a[1,2] <- 0.5

for(j in 1:rep) {
  #rho[j] <- a[j,1]
  rho[j] <- 0.1  
  lambda[j] <- a[j,2]
  
  ########Creating the Covariance Matrix
  for(m in 1:nt) {
    for(p in 1:nt) {
      t <- floor((m-1)/4)+1
      tau <- floor((p-1)/4)+1
      s <- abs(tau-t)
      if ((m-4*(t-1))==(p-4*(tau-1))) {
        i <- 1
      } else {
        i <- 0
      }
      if (s==0) {
        k <- 0
      } else {
        k <- 1
      }
      V[m,p] <- rho[j]^s/(1-rho[j]^2)+((1-k)*i + k* (rho[j]^s/n) + (rho[j]^(s+2) / (n * (1-rho[j]^2)) ) )*lambda[j]
      dVdrho [m,p] <- (rho[j]^(s-1) / (n * (rho[j]^2 -1)^2)) * ( (s*k+2*rho[j]^2+s*rho[j]^2-s*rho[j]^4-2*s*k*rho[j]^2+s*k*rho[j]^4)*lambda[j] + (2*n*rho[j]^2+n*s-n*s*rho[j]^2))
      dVdlambda[m,p] <- (1-k)*i + k* (rho[j]^s/n) + (rho[j]^(s+2) / (n * (1-rho[j]^2)) ) 
    }
  }
  
  #### Estimation
  
  inv_V <- solve (V)
  XtVinvX <- t(X)%*%inv_V%*%X
  inv1 <- solve(XtVinvX)
  beta <- inv1%*%t(X)%*%inv_V%*%e
  vtilde <- e-X%*%beta
  sigma2[j] <- as.numeric((1/nt)*(t(vtilde))%*%inv_V%*%(vtilde))
  ln_det_inv_V <- as.numeric(determinant(inv_V) [1])  # don't think this is used anywhere, replaced with the next line
  ln_det_V <- as.numeric(determinant(V) [1])  # where's the log?
  lnL_first_term[j] <- (nt/2)*(log(2*pi+log(sigma2[j])))  # is the 2pi supposed to be in a log by itself?
  lnL_second_term_second[j] <- as.numeric(t(vtilde)%*%inv_V%*%vtilde)
  lnL_second_term[j] <- (1/(2*sigma2[j]))*lnL_second_term_second[j]
  lnL_third_term[j] <- (1/2)*ln_det_V
  lnL[j] <- lnL_first_term[j]-lnL_second_term[j]-lnL_third_term[j]  # is there a (-) missing on the first term?
  
  grad[j, 1] <- (1/(2*sigma2[j]))*t(vtilde)%*%inv_V%*%dVdrho%*%inv_V%*%vtilde - sum(diag(dVdrho%*%inv_V))
  grad[j, 2] <- (1/(2*sigma2[j]))*t(vtilde)%*%inv_V%*%dVdlambda%*%inv_V%*%vtilde - sum(diag(dVdlambda%*%inv_V))
  
  if (j==1){
    gradient <- as.matrix(grad[j,])
    parameters <- as.matrix(a[j,])
  }else {
    gradient <- as.matrix(grad[j,] - grad[j-1,])
    parameters <- as.matrix(a[j,] - a[j-1,])   # a is never updated... maybe just searching over values in a?
  }
  
  step[j] <- abs(as.numeric(t(parameters)%*%(gradient)))/(norm(gradient))^2
  accuracy[j] <- norm(parameters)
  print(paste('Step:', j, '   Accuracy:', accuracy[j], '   LL:', lnL[j], '   lambda:', lambda[j]))
  
}

