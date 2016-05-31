import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


data = np.loadtxt("Model_data/run_train,tag_cross_entropy_loss-.csv", delimiter=',', skiprows=1, usecols=(1,2))

plt.title("Training Cross-Entropy Loss")
plt.plot(data[:,0],data[:,1])

plt.savefig("train_plot.png",format="png")