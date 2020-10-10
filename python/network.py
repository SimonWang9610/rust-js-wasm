import numpy as np
import json
from layer import Layer
from utils import relu_derivate

def optimize(layer, DZ, DW, alpha, input):
    layer.weights -= alpha * DW / input.shape[0]
    layer.bias -= alpha * np.sum(DZ, axis=1, keepdims=True) / input.shape[1]
    
def load(path):
    network = []
    with open(path, 'r') as f:
        data = json.load(f)
    
    for value in data.values():
        layer = Layer(value["neurons"], value["prev"], value["end"])
        layer.initial_parameters(np.array(value["weights"]), np.array(value["bias"]))
        network.append(layer)
        
    return NeuralNetwork(network)

def train(network, alpha, epoches, images_train, labels_train):
    
    for epoch in range(epoches):
        network.forward(images_train.T)
        loss = network.compute_loss(labels_train)
        accuracy = network.evaluate(labels_train)
        network.backward(alpha, labels_train)
        
        print(f'Epoch #{epoch}#: Loss: {loss}, Accuracy: {accuracy}')
        
        
class NeuralNetwork(object):
    
    def __init__(self, layers):
        self.layers = layers
        self.outputs = None
        
    def forward(self, input):
        outputs = [input]
        
        for i in range(len(self.layers)):
            output = self.layers[i].forward(outputs[-1])
            outputs.append(output)
        
        self.outputs = outputs
        
    def backward(self, alpha, target):
        # target [10, sample]
        weight = self.layers[-1].weights
        DZ = None
        DW = None
        
        for layer in self.layers[::-1]:
            output = self.outputs.pop()
            input = self.outputs[-1]
            
            if layer.end:
                DZ = (output - target) / target.shape[0]
            else:
                x = layer.weights.dot(input) + layer.bias
                DZ = weight.T.dot(DZ) * relu_derivate(x)
                weight = layer.weights
                
            DW = DZ.dot(input.T)
            optimize(layer, DZ, DW, alpha, input.T)
            
    def evaluate(self, target):
        # output [10, sample]
        # target [10, sample]
        output = self.outputs[-1]
        total = target.shape[1]
        correct = 0
        
        for i in range(output.shape[1]):
            if np.argmax(output[:, i:i+1]) == np.argmax(target[:, i:i+1]):
                correct += 1
                
        return correct / total
        
    def compute_loss(self, labels):
        # labels [10, sample]
        return - 1 / labels.shape[1] * np.sum(labels * np.log(self.outputs[-1]))
    
    def predict(self, input):
        output = input
        result =[]
        
        for layer in self.layers:
            output = layer.forward(output)
        
        for i in range(output.shape[1]):
            result.append(np.argmax(output[:, i:i+1]))
            
        return result
    
    def save(self):
        parameters = {}
        for i in range(len(self.layers)):
            layer_json = {
                "neurons": self.layers[i].neurons,
                "prev": self.layers[i].prev,
                "bias": self.layers[i].bias.tolist(),
                "weights": self.layers[i].weights.tolist(),
                "end": self.layers[i].end
            }
            parameters.update({i:layer_json})
            
        with open('parameters.json', 'w', encoding='utf-8') as f:
            json.dump(parameters, f, ensure_ascii=False, indent=4)

