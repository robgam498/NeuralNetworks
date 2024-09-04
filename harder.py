import numpy as np

# sets nonlinearity and maps s shaped curve with values between 0-1
def nonlin(x, deriv=False):
    if deriv:
        return x * (1-x)
    return 1/(1+np.exp(-x))

# inputs
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])

# outputs
y = np.array([ [0],[1],[1],[0] ])

np.random.seed(1)

# weights between the input layer l0/l1 and hidden layer l1/l2
syn0 = 2*np.random.random((3,4)) -1
# dimensions 3,4 = 3 inputs and 4 neurons in hidden layer
# weight matrix for 3 rows (3 nodes) and 4 columns (4 neurons)
# each element connects an input node to a hidden neuron

syn1 = 2*np.random.random((4,1)) -1
# dimensions 4,1 = 4 neurons and 1 output
# hidden layer has 4 neurons and output layer has 1 neuron (prediction 0,1)
# each element in this vector is a weight connecting a hidden neuron to output neuron


for j in range(60000):

    # feed in layers 0, 1, 2
    l0 = X
    # input layer
    l1 = nonlin(np.dot(l0,syn0))
    # hidden layer (dot product of l0 and syn0 passed through sigmoid function
    l2 = nonlin(np.dot(l1,syn1))
    # dot product of l1 and syn1 then passed through sigmoid function

    #how much did we miss the target value y by
    l2_error = y - l2
    # difference between predicted output l2 and actual output

    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))

    # in what direction is target value?
    l2_delta = l2_error*nonlin(l2,deriv=True)
    # error of output layer scaled by derivative
    # --> how much to adjust weights of connections between hidden and output layer

    # how much did each l1 value contribute to l2 error
    l1_error = l2_delta.dot(syn1.T)
    # error of hidden layer
    # propogating l2_delta backwards through the network

    #in what direction is target l1?
    l1_delta = l1_error*nonlin(l1,deriv=True)
    # error of hidden layer scaled by the derivative
    # --> how much to adjust weights of connections between input and hidden layer

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    # weights updated based on deltas, reducing error of network

# over time the network learns the pattern in the data and weights are adjusted to reduce prediction error (l2_error)