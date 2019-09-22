import matplotlib.pyplot as plt
import numpy as np

# create input signal
x = np.arange(-8, 8, 0.1)

# define weight [large weight is selected since we saw that increasing the weight makes the sigmoid to approach as step function]
w = 5.0

# define bias `b` [does not affect how quickly it ramps, rather it affects - which signal it activates for i.e. it moves the same slope along the x-axis]
# think of it like how much the signal of `b` drowns out the signal of `w`
b1 = -8.0
b2 = 0.0
b3 = 8.0

# define plot labels
l1 = 'b = -8.0'
l2 = 'b = 0.0'
l3 = 'b = 8.0'

# plot the data
for w, l, b in [(w, l1, b1), (w, l2, b2), (w, l3, b3)]:
    sigmoidfunction = 1 / (1 + np.exp((-x * w) + b))
    plt.plot(x, sigmoidfunction, label=l)

# label the plot
plt.xlabel('x')
plt.ylabel('h_wb(x)')  # activation function is now composed of `weights + bias + signal`
plt.legend(loc=2)
plt.show()
