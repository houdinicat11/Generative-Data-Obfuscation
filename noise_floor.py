#This script estimated the noise floor for a rho theta combination over 100 samples
import torch
from pyemd import emd_samples
import numpy as np
import os
import math
from torch.distributions.multivariate_normal import *
from torch.autograd.variable import Variable
from generator import *

#change the rho, and theta value's to calculate the sample earthmover distance of a certain combo of the two
#possible rho values: 1.5, 2, 2.5, 3
#possible theta values: 15, 30, 45
rho = 3
theta = 45

#size is the number of samples thrown into the EMD calculation from pyemd
size = 10000
#emdists is an array of 100 2 Wasserstein distances (Earthmover) estimated from 10,000 samples for each target population sample
noiseFloor = np.array(np.empty(1))

#calculate the covariance matrix values
t = math.radians(int(theta))
v1 = float(rho-0.5*pow(np.sin(2*t),2)*(rho-(1/rho)))
v2 = float(rho+0.5*pow(np.sin(2*t),2)*(rho-(1/rho)))
cv12 = float(-0.25*np.sin(4*t)*(rho-(1/rho)))
#set the covariance matrix values
sigma_P = torch.tensor([[v1,cv12], [cv12,v2]])
#set the target population
m = MultivariateNormal(torch.zeros(2), sigma_P)
#generate "size" samples from the target population

#sums the noise floor of the samples
for i in range(100):
  #generate sample from target
  target_sample_1 = Variable(m.sample([size]))
  target_sample_2 = Variable(m.sample([size]))

  noiseFloor[0] = noiseFloor[0] + emd_samples(target_sample_1.numpy(), target_sample_2.numpy())

#averages the noise floor
noiseFloor[0] = noiseFloor[0]/100.0

# result directory
result_path = "results/population_obfuscation/noise/theta-" + str(theta) + "/rho-" + str(rho) + "/"
#result file name
result_name = "NoiseFloor_rho-" + str(rho) + "_theta-" + str(theta) + ".txt"

#save the results
np.savetxt(result_path + result_name, noiseFloor)

