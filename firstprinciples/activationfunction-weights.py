import matplotlib.pyplot as plt
import numpy as np

# create input signal
x = np.arange(-8, 8, 0.1)

# define weights [which affects the slope i.e. how quickly it ramps/activates even for the right signal x]
w1 = 0.5
w2 = 1.0
w3 = 2.0

# define plot labels
l1 = 'w = 0.5'
l2 = 'w = 1.0'
l3 = 'w = 2.0'

# plot the data
for w, l in [(w1, l1), (w2, l2), (w3, l3)]:
    sigmoidfunction = 1 / (1 + np.exp(-x * w))
    plt.plot(x, sigmoidfunction, label=l)

# label the plot
plt.xlabel('x')
plt.ylabel('h_w(x)')
plt.legend(loc=2)
plt.show()
