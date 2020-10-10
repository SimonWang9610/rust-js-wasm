import numpy as np
from utils import relu, softmax

class Layer(object):
    def __init__(self, prev, neurons, end):
        self.neurons = neurons
        self.prev = prev
        self.end = end
        
        self.weights = None
        self.bias = None
    
    def initial_parameters(self, weights, bias):
        if weights is None:
            self.weights = np.random.randn(self.neurons, self.prev) * np.sqrt(1 / self.prev)
        else:
            self.weights = weights;
            
        if bias is None:
            self.bias = np.zeros((self.neurons, 1))
        else:
            self.bias = bias
            
    def forward(self, input):
        if self.weights is None:
            raise Exception("Weights no initialization!")
        
        z = self.weights.dot(input) + self.bias
        
        if self.end:
            return softmax(z)
        else:
            return relu(z)
        