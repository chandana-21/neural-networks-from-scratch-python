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


"""
Representation (4 inputs, 3 neurons)
   x=a[0]    a[1]

    O
               O 
    O 
               O 
    O
               O
    O
"""

print("Basic: ", output)

# •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••

# Vectorized

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

import numpy as np

output = np.dot(weights, inputs) + biases

print("Vectorized: ", output)

# •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••

# Multiple Inputs (3 examples)

inputs_matrix = [[1,2,3,2.5],
                 [2.0,5.0,-1.0,2.0],
                 [-1.5,2.7,3.3,-0.8]]

output_matrix = np.dot(inputs_matrix, np.array(weights).T) + biases

print("Matrix: ",output_matrix)

# •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••

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

# •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••

"""
np.random.seed(0) may not give the same results in jupyter nb etc as datatype might change during the
calculations, so nnfs.init() does the seed work along with setting a default datatype for numpy use
"""
import nnfs

nnfs.init()


# Using external dataset

from nnfs.datasets import spiral_data

X, y = spiral_data(100,3) # (samples, classes)
# X (100 samples, 2 input features)

layer_1_new = Layer_Dense(2,5)

# •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••

# Integrating Hidden Layer Activation Functions

"""
• We use Activation Functions as without it we only will be able to compute linear outputs even for 
non linear data, we'll only be able to fit the model using a linear function which is not efficient.

• Sigmoid has vanishing gradient problem.

• Reasons we use ReLU:
    1) It's granular
    2) Very fast 
"""

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)

activation_1 = Activation_ReLU()
layer_1_new.forward(X)
activation_1.forward(layer_1_new.output)

print("ReLU: ", activation_1.output[:5])

# •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••

# Softmax Activation

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

dense_1 = Layer_Dense(2,3)
activation_1 = Activation_ReLU()

dense_2 = Layer_Dense(3,3)
activation_2 = Activation_Softmax()

dense_1.forward(X)
activation_1.forward(dense_1.output)

dense_2.forward(activation_1.output)
activation_2.forward(dense_2.output)

print("Softmax: ", activation_2.output[:5])

# •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••

# Implementing Loss

class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # clipping so that -log(0) situation does not occur
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) 

        # This means that the person has passed scaler values and not the one-hot encoded values
        if len(y_true.shape) == 1: 
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # one-hot encoded vectors passed here
        elif len(y_true.shape) == 2: 
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation_2.output, y)

print("Loss: ", loss)

# •••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••

#  Backpropagation

