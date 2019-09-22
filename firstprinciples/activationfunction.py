import matplotlib.pyplot as plt
import numpy as np

# create input signal
x = np.arange(-8, 8, 0.1)

# create activation function that mimics f(x) = 1/(1 + e^(-x))
sigmoidfunction = 1 / (1 + np.exp(-x))

# plot the data
plt.plot(x, sigmoidfunction)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
