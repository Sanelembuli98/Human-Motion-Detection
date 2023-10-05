# Import necessary libraries
from layer import Layer  # Assuming you have a 'Layer' class defined in 'layer.py'
import numpy as np

# Create a class 'Dense' that inherits from the 'Layer' class


class Dense(Layer):
    # Constructor method for initializing the Dense layer
    def __init__(self, input_size, output_size):
        # Initialize the weights with random values and the bias with zeros
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    # Forward pass method for the Dense layer
    def forward(self, input):
        # Store the input for later use during backpropagation
        self.input = input
        # Perform matrix multiplication between weights and input, and add the bias
        return np.dot(self.weights, self.input) + self.bias

    # Backward pass method for the Dense layer
    def backward(self, output_gradient, learning_rate):
        # Calculate the gradient of the loss with respect to the weights
        weights_gradient = np.dot(output_gradient, self.input.T)
        # Update the weights using gradient descent
        self.weights -= learning_rate * weights_gradient
        # Update the bias using gradient descent
        self.bias -= learning_rate * output_gradient
        # Calculate the gradient of the loss with respect to the input
        return np.dot(self.weights.T, output_gradient)
