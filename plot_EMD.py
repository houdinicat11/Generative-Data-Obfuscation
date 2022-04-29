#Plots the mean 2 wasserstein distance of the Vanilla and Obfuscatory GAN on a log-log scale with a fitted line
import numpy as np
import matplotlib.pyplot as plt

#set a rho and theta value that you want to plot
#change rho and theta to another from the list below to plot a different rho theta combo
#possible theta values: 15, 30, 45
#possible rho values: 1.5, 2, 2.5, 3
rho = "2"
theta = "30"
#set the root directory for both the "noise" (Vanilla) and "marked" (Obfuscatory) GAN samples
noiseroot = "results/population_obfuscation/noise/theta-" +  theta + "/rho-" + rho + "/"
markedroot =  "results/population_obfuscation/marked/theta-" +  theta + "/rho-" + rho + "/"
#labels for the x axis
xLabels = ["Size 10", "Size 20", "Size 50", "Size 100"]
#x-values to bind them to
x = np.array([10,20,50,100])
#arrays to hold the mean 2 wasserstein distance for samples from the different GAN arcatextures
noise = np.array(np.empty(4))
marked = np.array(np.empty(4))

#load the Vanilla GAN generated EMD's
nsize10 = np.loadtxt(noiseroot + "size-10/SampleEMD_noise_rho-" + rho + "_theta-" + theta + "_size-10.txt")
nsize20 = np.loadtxt(noiseroot + "size-20/SampleEMD_noise_rho-" + rho + "_theta-" + theta + "_size-20.txt")
nsize50 = np.loadtxt(noiseroot + "size-50/SampleEMD_noise_rho-" + rho + "_theta-" + theta + "_size-50.txt")
nsize100 = np.loadtxt(noiseroot + "size-100/SampleEMD_noise_rho-" + rho + "_theta-" + theta + "_size-100.txt")

#average each result
noise[0] = np.mean(nsize10)
noise[1] = np.mean(nsize20)
noise[2] = np.mean(nsize50)
noise[3] = np.mean(nsize100)

#load the Obfuscatory GAN generated EMD's
msize10 = np.loadtxt(markedroot + "size-10/SampleEMD_marked_rho-" + rho + "_theta-" + theta + "_size-10.txt")
msize20 = np.loadtxt(markedroot + "size-20/SampleEMD_marked_rho-" + rho + "_theta-" + theta + "_size-20.txt")
msize50 = np.loadtxt(markedroot + "size-50/SampleEMD_marked_rho-" + rho + "_theta-" + theta + "_size-50.txt")
msize100 = np.loadtxt(markedroot + "size-100/SampleEMD_marked_rho-" + rho + "_theta-" + theta + "_size-100.txt")

#average each result
marked[0] = np.mean(msize10)
marked[1] = np.mean(msize20)
marked[2] = np.mean(msize50)
marked[3] = np.mean(msize100)

#load the noise floor
noiseFloor = np.loadtxt(noiseroot + "NoiseFloor_rho-" + rho + "_theta-" + theta + ".txt")

#plots the marked and noise
plt.scatter(x, noise)
plt.scatter(x, marked)
#plot the noise floor
#plt.axhline(y=noiseFloor, color="black", linestyle="-")

#get fitted lines through the noise and marked
a, b = np.polyfit(x, noise, 1)
c, d = np.polyfit(x, marked, 1)
#plot the fitted lines
plt.plot(x, a*x+b, 'blue')
plt.plot(x, c*x+d, 'orange')

#add a legend
plt.legend(labels=["Vanilla GAN", "Obfuscatory GAN"])
#makes an axis i guess (it just workes)
ax = plt.gca()
#sets the legend colors
leg = ax.get_legend()
leg.legendHandles[0].set_color('blue')
leg.legendHandles[1].set_color('orange')

#Sets the y-axis to log scale
plt.yscale('log')

#Binds the labels to the ticks
plt.xticks(x,xLabels)
#Titles the plot
plt.title("Log 2 Wasserstein Distance Theta=" + theta + ", Rho=" + rho)
#plt.ylim(-1,1.5)
plt.show()

#uncomment below to get the R-squared value
#coef = np.polyfit(x, noise, 1)
#yfit = np.polyval(coef, x)

#SStot = sum((noise - np.mean(noise))**2)
#SSres = sum((noise - yfit)**2)
#Rsq = 1-SSres/SStot
#print(Rsq)