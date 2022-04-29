#this files loads the generator for a theta, rho, size combo and generates samples of a certain sizes for each genertaor and saves them in the fake_data directory
import os
import pandas as pd
from generator import *

#set the size of the sample
size = 500
#set the base directory
base_dir = "models/population_obfuscation/marked/generator/theta-15/rho-2.5/size-100/"
i = 0

#loop through the .pt files
for root, dirs, files in os.walk(base_dir):
  for file in files:
    #get rho, theta, size values
    file_info = file.split('_')
    gen_from = file_info[0]
    rho = float(file_info[1].split('-')[-1])
    theta = file_info[2].split('-')[-1]
    target_size = file_info[3].split('-')[-1]
    sample = file_info[4].split('-')[-1].split('.')[0]

    #integer rho cases
    if rho > 1.9 and rho < 2.1:
      rho = 2
    elif rho > 2.9:
      rho = 3

    #gen_from set
    if gen_from == "marked":
      gen_from_marked = True
    else:
      gen_from_marked = False

    #load and initilize the generator
    checkpoint= torch.load("models/population_obfuscation/" + gen_from + "/generator" + "/theta-" + str(theta) + "/rho-" + str(rho) + "/size-" + str(target_size) + "/" + file, map_location='cpu')
    generator = GeneratorNet(2, float(rho), gen_from_marked)
    generator.load_state_dict(checkpoint['model_state_dict'])
    #generate new data
    new_data = generator.create_data(size)

    #turn data into a dataframe
    df = pd.DataFrame(new_data, columns=['V1', 'V2'])
    #Changes the name to be easier to read
    output_dir = "theta-" + theta + "/rho-" + str(rho) + "/size-" + target_size + "/"
    output_name = gen_from + "_rho-" + str(rho) + "_theta-" + theta + "_size-" + target_size + "_sample-" + sample
    #output data into the fake_data directory
    df.to_csv("fake_data/population_obfuscation/" + gen_from + "/" + output_dir + output_name + ".csv", index=False)