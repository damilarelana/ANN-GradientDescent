import numpy as np
import timeit


# create input signal
x = np.arange(-8, 8, 0.1)

# Define the weight matrices across layers
W1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])  # refers to weights applied to Layer 1
W2 = np.zeros((1, 3))  # helps to ensure we set it as an ndarray of rank 2 i.e. 2 dimensions [which would not be the case if np.array([0.5, 0.5, 0.5]) is assigned directly]
W2[0, :] = np.array([0.5, 0.5, 0.5])  # assigning a value only to the first row [as there is only one output node]


# define bias `b` [does not affect how quickly it ramps, rather it affects - which signal it activates for i.e. it moves the same slope along the x-axis]
# think of it like how much the signal of `b` drowns out the signal of `w`
b1 = np.array([0.8, 0.8, 0.8])  # `0.8` is used because `activationfunction-weights-bias.py` shows `activation slope` for 'b = 0` is slightly shifted to left of `x=0`
b2 = np.array([0.2])  # b2 refers to bias applied to layer 2


# define the signmoid function
def sigmoidfunction(input):
    return 1 / (1 + np.exp(-input))


# define the neural network
def simple_neural_network(num_of_layers, input_signal, weights, bias):
    for layer_num in range(num_of_layers - 1):
        if layer_num == 0:
            input_to_next_node = input_signal
        else:
            input_to_next_node = output_signal  # where 'h' is defined as standard output signal at each layer from `layer 2`

        # define the output_signal array
        output_signal = np.zeros((weights[layer_num].shape[0],))  # output signal structure would have same number of rows as the weight

        # Loop through the weights values in the columns of each row
        for output_node_num in range(weights[layer_num].shape[0]): 
            sigmoid_function_input_value = 0  # initialize the sigmoidfunction value for that weight array
            for input_node_num in range(weights[layer_num].shape[1]):  # the row-wise weights within that weight array
                sigmoid_function_input_value += weights[layer_num][output_node_num][input_node_num] * input_to_next_node[input_node_num]

            # add the bias
            sigmoid_function_input_value += bias[layer_num][output_node_num]

            # calculate the sigmoid value
            output_signal[output_node_num] = sigmoidfunction(sigmoid_function_input_value)
    return output_signal


#  define the weights, input signals, bias etc.
weights = [W1, W2]
bias = [b1, b2]
input_signal = [1.5, 2.0, 3.0]


# run neural network
#  - `number` times
#  - output the `time` it takes to complete it
print('Output:', simple_neural_network(3, input_signal, weights, bias))
print(timeit.timeit("simple_neural_network(3, input_signal, weights, bias)", number=1000000, globals=globals()))
