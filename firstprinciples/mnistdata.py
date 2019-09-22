from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import timeit
from sklearn.metrics import accuracy_score


# initialize
digits = load_digits()

# normalize the data
normalizer = StandardScaler()
normalized_input = normalizer.fit_transform(digits.data)

# create training and test data
expected_output = digits.target
input_training_data, input_test_data, expected_training_output, expected_test_output = train_test_split(normalized_input, expected_output, test_size=0.4)


# function to convert the expected output into a sparse matrix, suitable as an ANN output
# while making each vector representing the digits [0 ... 9]
# placing the digits at index i,i in the new expected output matrix
def vectorizer(expected_output):
    expected_output_vector = np.zeros((len(expected_output), 10))
    for i in range(len(expected_output)):
        expected_output_vector[i, expected_output[i]] = 1
    return expected_output_vector


# vectorize the data
vectorized_training_output = vectorizer(expected_training_output) 
vectorized_test_output = vectorizer(expected_test_output)

# show
# print(digits.data.shape)
# plt.gray()
# plt.matshow(digits.images[2])
# plt.show()
# # print(digits.data[2, :])  # show one of the unnormalized data value

# # show the one of the normalized data value
# print(normalized_input.data.shape)
# print(normalized_input[2, :])   # show one of the normalized data value

# print(expected_training_output[0])
# print(vectorized_training_output[0])


# define the sigmoid function
def sigmoidfunction(input):
    return 1 / (1 + np.exp(-input))


# define derivative of the sigmoid function
def sigmoidfunction_derivative(input):
    return sigmoidfunction(input) * (1 - sigmoidfunction(input))


# define the neural network layer structure
# 64 input nodes [equivalent to the 64 elements of each of the image data elements]
# 30 hidden layer nodes
# 10 output nodes [equivalent to the 10 digits]
neuralnetwork_layer_structure = [64, 30, 10]


# define the structure of weights and biases
def initialize_weights(neuralnetwork_layer_structure):
    W = {}  # initialize as an empty dictionary
    b = {}  # initialize as an empty dictionary

    # populate the weights for each layer, using the layer structure defined earlier
    # remember that for `n` layers, there are usually `n-1` number of weights
    for layer_num in range(1, len(neuralnetwork_layer_structure)):  # layer numbers do not start from zero, hence initial index '1'
        W[layer_num] = np.random.random_sample((neuralnetwork_layer_structure[layer_num], neuralnetwork_layer_structure[layer_num-1]))
        b[layer_num] = np.random.random_sample((neuralnetwork_layer_structure[layer_num], ))
    return W, b


# define the structure of partial derivative summation
# initially these would be zero value before we start training the neural network
def initialize_weights_deltas(neuralnetwork_layer_structure):
    delta_W = {}
    delta_b = {}

    for layer_num in range(1, len(neuralnetwork_layer_structure)):
        delta_W[layer_num] = np.zeros((neuralnetwork_layer_structure[layer_num], neuralnetwork_layer_structure[layer_num-1]))
        delta_b[layer_num] = np.zeros((neuralnetwork_layer_structure[layer_num], ))
    return delta_W, delta_b


# define the feedforward
def feedforward(input_data, W, b):
    h = {1: input_data}
    z = {}

    for layer_num in range(1, len(W) + 1):
        if layer_num == 1:
            input_to_next_layer = input_data
        else:
            input_to_next_layer = h[layer_num]

        # using numpy's element-wise matrix multiplication
        z[layer_num + 1] = W[layer_num].dot(input_to_next_layer) + b[layer_num]  # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[layer_num + 1] = sigmoidfunction(z[layer_num + 1])

    return h, z


# define the output layers dirac value
# d^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
def calc_output_layer_dirac(output, h_out, z_out):
    return -(output - h_out) * sigmoidfunction_derivative(z_out)


# define the hidden layers dirac value
# d^(l) = (transpose(W^l)) * dirac^(l+1) * f'(z^(l))
def calc_hidden_layer_dirac(dirac_plus_1, w_l, z_l):
    return np.dot(np.transpose(w_l), dirac_plus_1) * sigmoidfunction_derivative(z_l)


# define the training neural network
def training_neuralnetwork(neuralnetwork_layer_structure, normalized_input, expected_output, max_iteration_count=20000, stepsize=0.25):
    W, b = initialize_weights(neuralnetwork_layer_structure)
    iteration_count = 0
    output_length = len(expected_output)
    avg_cost_function = []  # yes the average cost function would be a vector

    print('Gradient descent initiated for {} iterations'.format(max_iteration_count))

    while iteration_count < max_iteration_count:
        if iteration_count % 1000 == 0:
            print('Iteration {} of {}'. format(iteration_count, max_iteration_count))
        delta_W, delta_b = initialize_weights_deltas(neuralnetwork_layer_structure)
        avg_cost = 0

        for i in range(len(expected_output)):
            dirac = {}

            # feedforward, pass stored h/z values to be used in the gradient descent step
            h, z = feedforward(normalized_input[i, :], W, b)

            # start backpropagating the errors via a forloop
            for layer_num in range(len(neuralnetwork_layer_structure), 0, -1):  # start counting from layers n down to layer 1
                if layer_num == len(neuralnetwork_layer_structure):  # i.e. if at output layer
                    dirac[layer_num] = calc_output_layer_dirac(expected_output[i,:], h[layer_num], z[layer_num])
                    avg_cost += np.linalg.norm(expected_output[i,:] - h[layer_num])
                else:
                    if layer_num > 1:
                        dirac[layer_num] = calc_hidden_layer_dirac(dirac[layer_num+1], W[layer_num], z[layer_num])

                    # deltaW^[l] = delta^W[l] + delta^(l+1) * transpose(h^(l))
                    delta_W[layer_num] += np.dot(dirac[layer_num+1][:,np.newaxis], np.transpose(h[layer_num][:,np.newaxis]))                    

                    # deltab^[l] = delta^b[l] + dirac(l+1)
                    delta_b[layer_num] += dirac[layer_num+1]

        # implement gradient descent for weights per layer
        for layer_num in range(len(neuralnetwork_layer_structure) - 1, 0, -1):
            W[layer_num] += -stepsize * (1.0/output_length * delta_W[layer_num])
            b[layer_num] += -stepsize * (1.0/output_length * delta_b[layer_num])

        # accumulate the average cost function for just 1 training data
        avg_cost = (1.0/output_length) * avg_cost

        # accummulate the average cost function vector over the entire training set
        avg_cost_function.append(avg_cost)

        # increase the iteration_count
        iteration_count += 1

    return W, b, avg_cost_function


# start training by passing the training data and/or time it
# print(timeit.timeit("W, b, avg_cost_function = training_neuralnetwork(neuralnetwork_layer_structure, input_training_data, vectorized_training_output)", number=1, globals=globals()))
W, b, avg_cost_function = training_neuralnetwork(neuralnetwork_layer_structure, input_training_data, vectorized_training_output)

# plot the results
plt.plot(avg_cost_function)
plt.ylabel('Average J')
plt.xlabel('Iteration Number')
plt.show()


# test the trained weights
def predicter(W, b, normalized_input, num_neuralnetwork_layers):
    sample_number_shape = normalized_input.shape[0]  # extract number of rows in sample number vector
    predicted_output = np.zeros((sample_number_shape, ))  # expected out should have same number of rows i.e. nodes i.e. [0 , 1, 0 ... 0]
    for i in range(sample_number_shape):  # range of numbers from 0 about 1700+ i.e. the inputs are stacked column-wise, with each input being a row vector of 64 elements
        h, z = feedforward(normalized_input[i, :], W, b)  # an h(3) is obtained for 64 element vector at index i
        predicted_output[i] = np.argmax(h[num_neuralnetwork_layers])  # extract index of node with max value in the 3rd layer
    return predicted_output


# compare the expected output to the predicted output
predicted_output = predicter(W, b, input_test_data, 3)
print('Accuracy:', accuracy_score(expected_test_output, predicted_output)*100)
