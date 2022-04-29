#I really don't use this a lot so I stopped updating it, I recomend using plot_EMD.py instead.
import numpy as np
import matplotlib.pyplot as plt


rho = "2"
theta = "15"
noiseroot = "results/population_obfuscation/noise/theta-" +  theta + "/rho-" + rho + "/"
markedroot =  "results/population_obfuscation/marked/theta-" +  theta + "/rho-" + rho + "/"
xLabels = ["Size 10", "Size 20", "Size 50", "Size 100"]
ticks = [0,2,4,6]
left_pos = [-0.4,1.6,3.6,5.6]
right_pos = [0.4,2.4,4.4,6.4]
leftdata = np.array(np.empty(4))
rightdata = np.array((4,100))


nsize10 = np.loadtxt(noiseroot + "size-10/noise_rho-" + rho + "_theta-" + theta + "_size-10.txt")
nsize20 = np.loadtxt(noiseroot + "size-20/noise_rho-" + rho + "_theta-" + theta + "_size-20.txt")
nsize50 = np.loadtxt(noiseroot + "size-50/noise_rho-" + rho + "_theta-" + theta + "_size-50.txt")
nsize100 = np.loadtxt(noiseroot + "size-100/noise_rho-" + rho + "_theta-" + theta + "_size-100.txt")

for i in range(nsize10.size):
  nsize10[i] = np.log(nsize10[i])
for i in range(nsize20.size):
  nsize20[i] = np.log(nsize20[i])
for i in range(nsize50.size):
  nsize50[i] = np.log(nsize50[i])
for i in range(nsize10.size):
  nsize100[i] = np.log(nsize100[i])
leftdata = [nsize10, nsize20, nsize50, nsize100]


msize10 = np.loadtxt(markedroot + "size-10/marked_rho-" + rho + "_theta-" + theta + "_size-10.txt")
msize20 = np.loadtxt(markedroot + "size-20/marked_rho-" + rho + "_theta-" + theta + "_size-20.txt")
msize50 = np.loadtxt(markedroot + "size-50/marked_rho-" + rho + "_theta-" + theta + "_size-50.txt")
msize100 = np.loadtxt(markedroot + "size-100/marked_rho-" + rho + "_theta-" + theta + "_size-100.txt")

for i in range(msize10.size):
  msize10[i] = np.log(msize10[i])
for i in range(msize20.size):
  msize20[i] = np.log(msize20[i])
for i in range(msize50.size):
  msize50[i] = np.log(msize50[i])
for i in range(msize100.size):
  msize100[i] = np.log(msize100[i])
rightdata = [msize10, msize20, msize50, msize100]

plt.boxplot(leftdata, positions=left_pos)
plt.boxplot(rightdata, positions=right_pos, patch_artist=True)
plt.legend(labels=["Noise", "Marked"])
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color('white')
leg.legendHandles[1].set_color('blue')

#Binds the labels to the ticks
plt.xticks(ticks,xLabels)
#Titles the plot
plt.title("Log 2 Wasserstein Distance Theta=" + theta + ", Rho=" + rho)
plt.show()
