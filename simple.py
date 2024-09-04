import numpy as np

# sigmoid function = mapping any value to a value between 0 and 1
# we use this to convert numbers to probabilities + provides nonlinearity
def nonlin(x, deriv=False):
    if deriv:
        return x * (1-x)
    return 1/(1+np.exp(-x))

# input dataset, 3 input nodes, 1 output, and 4 training examples
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])

# output dataset, horizontal 1 row and 4 columns
# T is transpose --> 4 rows, 1 column
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculations deterministic
np.random.seed(1)

# initialize weights randomly w/ mean 0
# syn0 = synapse0, we use 3,1 because 3 inputs 1 output
# l0 is of size 3, l1 is of size 1 --> (3,1) is required to connect every node in l0 to l1
syn0 = 2*np.random.random((3,1)) - 1

for iter in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    # l1 represents the network's guess/prediction after processing

    # how much did we miss?
    l1_error = y-l1

    # multiplying how much we missed BY the slope of the sigmoid at values in l1
    l1_delta = l1_error * nonlin(l1,True)
    # l1_error is showing how much prediction (l1) is off from actual output y
    # nonlin tells us how sensitive the output of sigmoid function is to changes in input at l1
    # l1_delta combines error and derivative to determine how much to adjust the weights

    # update weights
    syn0 += np.dot(l0.T,l1_delta)
    # np.dot calculates the gradient (change) of weights needed to reduce error, transposing input l0 and l1_delta
    # syn0 += updates weights by adding to the gradient, over many iterations this will make better predictions

print("Output After Training")
print(l1)

