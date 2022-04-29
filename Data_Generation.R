################################################################################
# Data Generation
library(readr)

# Generate the Marked Popoulation
my_nP <- 10000  # Specify sample size --  very large, assume infinite
my_mu1 <- c(0,0) # Specify the means of the variables

rho  <- 3
var1 <- rho
var2 <- 1/rho

# Generate variance/covariance matrix of population
my_SigmaP <- matrix(c(var1, 0, 0, var2), byrow=TRUE,ncol=2)

# Generate Population sample
library(MASS)
YP <- mvrnorm(n = my_nP, mu = my_mu1, Sigma = my_SigmaP)  

#  Generate the sample from the target population
nT = 100 # sample size
theta = pi/4 # rotation angle

# Gernerate variance/covariance matrix
v1 = var1-0.5 * sin(2*theta)^2*(var1-var2)
v2 = var1+0.5 * sin(2*theta)^2*(var1-var2)
cv12 = -0.25*sin(4*theta)*(var1-var2)
my_SigmaT <- matrix(c(v1, cv12, cv12, v2), byrow=TRUE,ncol=2)
 

#write.csv(YP, "C:/Users/User/Documents/Math_Research/population_sample/pop_sample_0.csv", sep=',', row.names = FALSE)

for(i in 1:100)
{
   YT <- mvrnorm(n=nT, mu = my_mu1, Sigma = my_SigmaT)
   write.csv(YT, paste0("C:/Users/User/Documents/Math_Research/GAN-generate-data-master/GAN-generate-data-master/original_data/population_obfuscation/target/theta-45/rho-",rho,"/size-",nT,"/target_rho-",rho,"_theta-45_size-",nT,"_sample-",i,".csv"), sep=',', row.names = FALSE)
}
