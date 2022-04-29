#Calculates the EMDists as defined in https://en.wikipedia.org/wiki/Wasserstein_metric Section "Examples" Subsection "Normal Distributions"
#ignore the error for discarding imaginary parts, the imaginary parts are always + 0i ie they dont exist.
import numpy as np
import os
import pandas as pd
import numpy.linalg as la
import scipy.linalg as sp
import math


#initilizes EMDist array
emdists = np.array(np.empty(100))
# vectors of means
M1 = np.matrix([[0],[0]])
M2 = np.matrix([[0],[0]])

#base directory for calculating the EMDists from
#possible rho values: 1.5, 2, 2.5, 3
#possible theta values: 15, 30, 45
#possible size values: 10, 20, 50, 100
#possible gen_from values: "marked", "noise"
base_dir = "GAN-generate-data-master/GAN-generate-data-master/fake_data/population_obfuscation/marked/theta-15/rho-2.5/size-100"
#error checking iterable
i = 0

#walks through all files in the directory
for root, dirs, files in os.walk(base_dir):
  for file in files:
    #get file info
    file_info = file.split('_')
    gen_from = file_info[0]
    rho = float(file_info[1].split('-')[-1])
    theta = file_info[2].split('-')[-1]
    target_size = file_info[3].split('-')[-1]
    t = math.radians(int(theta))

    #read in the csv
    df = pd.read_csv(root + "/" + file)
    data = np.array(df.values)
    M2[0] = np.mean(data[:,0])
    M2[1] = np.mean(data[:,1])

    #The population cov matrix
    v1 = rho-0.5*pow(np.sin(2*t),2)*(rho-(1/rho))
    v2 = rho+0.5*pow(np.sin(2*t),2)*(rho-(1/rho))
    cv12 = -0.25*np.sin(4*t)*(rho-(1/rho))
    C1 = np.matrix(((v1,cv12),(cv12,v2)))
    #The Sample cov matrix
    C2 = np.cov(np.transpose(data))

    # distance between the means
    location = pow(la.norm(M2 - M1),2)
    size = np.trace(C2) + np.trace(C1)
    shape = np.trace(2*sp.sqrtm((np.matmul(np.matmul(sp.sqrtm(C1+0j),C2),sp.sqrtm(C1+0j)+0j))))
    emdists[i] = location + size - shape

    i = i + 1

print(i)

#integer rho check
if rho > 1.9 and rho < 2.1:
  rho = 2
elif rho > 2.9:
  rho = 3

#save directory
result_path = "GAN-generate-data-master/GAN-generate-data-master/results/population_obfuscation/" + gen_from + "/theta-" + str(theta) + "/rho-" + str(rho) + "/size-" + str(target_size) + "/"
#save file name
result_name = gen_from + "_rho-" + str(rho) + "_theta-" + str(theta) + "_size-" + str(target_size) + ".txt"

#save the distances
np.savetxt(result_path + result_name, emdists)

