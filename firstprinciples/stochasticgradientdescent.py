import numpy as np

# Given the polynomial function f(x) = X^4 - 3x^3 + 2
# we want to determine the value of x such that
# error (i.e. f(oldX) - f(newX) ) is minimised i.e. derror/dx is `0` when we plot `error` against `x`
# stochastic gradient descent suggests to us that
# newX = oldX - (step*derror/dx)
oldX = np.float64(0.0)
newX = np.float64(6.0)
stepValue = np.float64(0.01)
precision = np.float64(0.00001)

print(type(newX))


# derivative of `f(x) = X^4 - 3x^3 + 2` is `4x^3 - 9x^2`
def derivative(x):
    y = np.poly1d([4, -9, 0, 0])
    return y(x)  # return (4 * (x**3)) - (9 * (x**2))


while abs(newX - oldX) > precision:
    oldX = newX
    newX += -stepValue * derivative(oldX)

print('The local minimum is {}'.format(newX))
