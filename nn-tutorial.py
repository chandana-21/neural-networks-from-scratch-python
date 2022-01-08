inputs = [1,2,3,2.5]

w1 = [0.2,0.8,-0.5,1.0]
w2 = [0.5,-0.91,0.26,-0.5]
w3 = [-0.26,-0.27,0.17,0.87]

b1 = 2
b2 = 3
b3 = 0.5

output = [inputs[0]*w1[0] + inputs[1]*w1[1] + inputs[2]*w1[2] + inputs[3]*w1[3] + b1,
          inputs[0]*w2[0] + inputs[1]*w2[1] + inputs[2]*w2[2] + inputs[3]*w2[3] + b2,
          inputs[0]*w3[0] + inputs[1]*w3[1] + inputs[2]*w3[2] + inputs[3]*w3[3] + b3
        ]

# Representation (4 inputs, 3 neurons)
#    x=a[0]    a[1]
# 
#     O
#                O 
#     O 
#                O 
#     O
#                O
#     O

print("Basic: ", output)

# Vectorized

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

import numpy as np

output = np.dot(weights, inputs) + biases

print("Vectorized: ", output)

# Multiple Inputs (3 examples)

inputs_matrix = [[1,2,3,2.5],
                 [2.0,5.0,-1.0,2.0],
                 [-1.5,2.7,3.3,-0.8]]

output_matrix = np.dot(inputs_matrix, np.array(weights).T) + biases

print("Matrix: ",output_matrix)

# Layers and Random Initialization of weights and biases

X = [[1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]

np.random.seed(0) # With the seed reset (every time), the same set of numbers will appear every time.


class Layer_Dense:

    def __init__(self,n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)*0.1
        self.biases = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer_1 = Layer_Dense(4,5) # (x_1, a_1) (x_1 = len(X[0]) == 4)
layer_2 = Layer_Dense(5,2) # (x_2 = a_1, a_2)

layer_1.forward(X)

print("Layer 1 output: ", layer_1.output)

layer_2.forward(layer_1.output)

print("Layer 2 output: ", layer_2.output)

# np.random.seed(0) may not give the same results in jupyter nb etc as datatype might change during the
# calculations, so nnfs.init() does the seed work along with setting a default datatype for numpy use

import nnfs

nnfs.init()

# Integrating Hidden Layer Activation Functions


# We use Activation Functions as without it we only will be able to compute linear outputs even for 
# non linear data, we'll only be able to fit the model using a linear function which is not efficient.

# Sigmoid has vanishing gradient problem.
# Reasons we use ReLU:
#     • It's granular
#     • Very fast 

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)

