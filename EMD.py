#This script calculates the sample 2 Wasserstein distance across all 100 samples for a given rho, theta, size, and gen_from value
#and saves it in the results directory. https://github.com/wmayner/pyemd
import torch
from pyemd import emd_samples
import numpy as np
import os
import math
from torch.distributions.multivariate_normal import *
from torch.autograd.variable import Variable
from generator import *

#change gen_from between "noise" (Vanilla GAN) and "marked" (Obfuscatory GAN) to calculate the 2 Wasserstein distance of that GAN
#possible gen_from values: "marked", "noise"
gen_from = "noise"

#change the rho, theta, and size value's to calculate the sample earthmover distance of a certain combo of the three
#possible rho values: 1.5, 2, 2.5, 3
#possible theta values: 15, 30, 45
#possible size values: 10, 20, 50, 100
rho = 2
theta = 30
target_size = 50

#size is the number of samples thrown into the EMD calculation from pyemd
size = 10000
#emdists is an array of 100 2 Wasserstein distances (Earthmover) estimated from 10,000 samples for each target population sample
emdists = np.array(np.empty(100))
#base directory of generator models, change marked to noise to get the generator files from the other type of GAN and vice versa
base_dir = "models/population_obfuscation/" + gen_from + "/generator/theta-" + str(theta) + "/rho-" + str(rho) + "/size-" + str(target_size) + "/"
#keeps track of how many 2 wasserstein distances were claculated for error checking
i = 0

#gen_from_marked set
if gen_from == "marked":
  gen_from_marked = True
else:
  gen_from_marked = False

#t is theta in radians
t = math.radians(int(theta))
#calculate the covariance matrix for the target values
v1 = float(rho-0.5*pow(np.sin(2*t),2)*(rho-(1/rho)))
v2 = float(rho+0.5*pow(np.sin(2*t),2)*(rho-(1/rho)))
cv12 = float(-0.25*np.sin(4*t)*(rho-(1/rho)))
#set the target covariance matrix values
sigma_P = torch.tensor([[v1,cv12], [cv12,v2]])
#set the target population
m = MultivariateNormal(torch.zeros(2), sigma_P)

#These for loops iterate through all the files under a given base directory
for root, dirs, files in os.walk(base_dir):
  for file in files:
    #generate sample from target
    target_sample = Variable(m.sample([size]))

    #generate sample from learned
    #load the .pt file
    checkpoint= torch.load("models/population_obfuscation/" + gen_from + "/generator" + "/theta-" + str(theta) + "/rho-" + str(rho) + "/size-" + str(target_size) + "/" + file, map_location='cpu')
    #get the generator parameters
    generator = GeneratorNet(2, float(rho), gen_from_marked)
    #load the weights
    generator.load_state_dict(checkpoint['model_state_dict'])
    #generate "size" sample from the learned distribution
    generated_sample = generator.create_data(size)

    #calculate the sample 2 Wasserstein distance between the target and learned distribution
    emdists[i] = emd_samples(target_sample.numpy(), generated_sample)
    #incriment i
    i = i + 1

print(i)

# result directory
result_path = "results/population_obfuscation/" + gen_from + "/theta-" + str(theta) + "/rho-" + str(rho) + "/size-" + str(target_size) + "/"
#result file name
result_name = "SampleEMD_" + gen_from + "_rho-" + str(rho) + "_theta-" + str(theta) + "_size-" + str(target_size) + ".txt"

#save the results
np.savetxt(result_path + result_name, emdists)

