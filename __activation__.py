# Import necessary libraries
from layer import Layer
import numpy as np
import sys
import os

# Create a class 'Activation' that inherits from the 'Layer' class


class Activation(Layer):
    # Constructor method for initializing the Activation layer
    def __init__(self, activation, activation_prime):
        # Store the activation function
        self.activation = activation
        # Store the derivative of the activation function
        self.activation_prime = activation_prime

    # Forward pass method for the Activation layer
    def forward(self, input):
        # Store the input for later use during backpropagation
        self.input = input
        # Apply the activation function to the input and return the result
        return self.activation(self.input)

    # Backward pass method for the Activation layer
    def backward(self, output_gradient, learning_rate):
        # Calculate the gradient of the loss with respect to the input
        # by multiplying the output gradient with the derivative of the activation function
        return np.multiply(output_gradient, self.activation_prime(self.input))
