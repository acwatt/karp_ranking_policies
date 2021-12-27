install.packages("dummies")
library(dummies)

data <- read.csv("G:/My Drive/PhD/GSR/Larry GSR/data/clean/grouped_nation.1751_2014.csv")

data<-subset(data, Year>= 1945 & Year<=2005)

data$time<- data$time-44

data$gp <- factor(data$group)

data <- data[order(data$time),]

as.numeric(data$gp)

b <- dummy(data$gp, sep = "_")

#x1 <- matrix(data = data$gp_BRIC, ncol = 1)
#x2 <- matrix(data = data$gp_EU, ncol = 1)
#x3 <- matrix(data = data$gp_Other, ncol = 1)
#x4 <- matrix(data = data$gp_USA, ncol = 1)

t <- matrix(data=data$time, ncol=1)

t2 <- t^2

X <- cbind(b,t,t2)

e <- matrix(data=data$CO2, ncol=1)

n <- 4
T <- 60
nt <- n*T
rep <- 40

V <- matrix(1:(nt*nt), nrow = nt)
dVdrho <- matrix(1:(nt*nt), nrow = nt)
dVdlambda <- matrix(1:(nt*nt), nrow = nt)
a <- matrix(1:(2*rep), nrow=rep)
rho <- matrix(1:rep, nrow = rep)
lambda <- matrix(1:rep, nrow = rep)
sigma2 <- matrix(1:rep, nrow = rep)
lnL_first_term <- matrix(1:rep, nrow = rep)
lnL_second_term_second <- matrix(1:rep, nrow = rep)
lnL_second_term <- matrix(1:rep, nrow = rep)
lnL_third_term <- matrix(1:rep, nrow = rep)
lnL <- matrix(1:rep, nrow = rep)
grad <- matrix(1:(2*rep), nrow = rep)
step <- matrix(1:rep, nrow = rep)
accuracy <- matrix(1:rep, nrow = rep)

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
ln_det_inv_V <- as.numeric(determinant(inv_V) [1])
ln_det_V <- as.numeric(determinant(V) [1])
lnL_first_term[j] <- (nt/2)*(log(2*pi+log(sigma2[j])))
lnL_second_term_second[j] <- as.numeric(t(vtilde)%*%inv_V%*%vtilde)
lnL_second_term[j] <- (1/(2*sigma2[j]))*lnL_second_term_second[j]
lnL_third_term[j] <- (1/2)*ln_det_V
lnL[j] <- lnL_first_term[j]-lnL_second_term[j]-lnL_third_term[j]

grad[j, 1] <- (1/(2*sigma2[j]))*t(vtilde)%*%inv_V%*%dVdrho%*%inv_V%*%vtilde - sum(diag(dVdrho%*%inv_V))
grad[j, 2] <- (1/(2*sigma2[j]))*t(vtilde)%*%inv_V%*%dVdlambda%*%inv_V%*%vtilde - sum(diag(dVdlambda%*%inv_V))

if (j==1){
gradient <- as.matrix(grad[j,])
parameters <- as.matrix(a[j,])
}else {
  gradient <- as.matrix(grad[j,] - grad[j-1,])
  parameters <- as.matrix(a[j,] - a[j-1,])
}

step[j] <- abs(as.numeric(t(parameters)%*%(gradient)))/(norm(gradient))^2
accuracy[j] <- norm(parameters)

}

