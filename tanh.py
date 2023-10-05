from activation import Activation
import numpy as np


class Tanh(Activation):
    def __init__(self):

        # Apply Hyperbolic Tangent to normalize the output
        # of a Neuron. This function introduces non-linearity
        # into the network.
        def tanh(x): return np.tanh(x)
        def tanh_prime(x): return 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)
