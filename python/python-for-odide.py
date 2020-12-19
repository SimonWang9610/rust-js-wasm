import numpy as np
from js import parameters, input

def relu(x):
    return (x >= 0).astype(int) * x

def softmax(x):
    temp = x - x.max()
    return np.exp(temp) / np.sum(np.exp(temp), axis=0, keepdims=True)

def load():
    network = []
    
    for i in range(3):
        layer = Layer(parameters[i].prev, parameters[i].neurons, parameters[i].end)
        layer.initial_parameters(np.asarray(parameters[i].weights, dtype=np.float32), np.asarray(parameters[i].bias, dtype=np.float32));
        network.append(layer)
    return NeuralNetwork(network)

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
            self.weights = weights.reshape((self.neurons, self.prev));
            
        if bias is None:
            self.bias = np.zeros((self.neurons, 1))
        else:
            self.bias = bias.reshape((self.neurons, 1))
            
    def forward(self, input):
        if self.weights is None:
            raise Exception("Weights no initialization!")
        
        z = self.weights.dot(input) + self.bias
        
        if self.end:
            return softmax(z)
        else:
            return relu(z)
        
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input):
        output = input.T
        result =[]
        
        for layer in self.layers:
            output = layer.forward(output)
        
        for i in range(output.shape[1]):
            result.append(np.argmax(output[:, i:i+1]))
            
        return result

nn = load()
predict = nn.forward(np.asarray(input, dtype=np.float32).reshape(10000, 784))
print(f'predict result : {predict}')
print(f'total sample: {len(predict)}')