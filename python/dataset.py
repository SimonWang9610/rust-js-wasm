from sklearn.datasets import fetch_openml
from network import NeuralNetwork, train
from utils import one_hot
from layer import Layer
 
mnist = fetch_openml('mnist_784', version=1)
print('load dataset...')
# [samples, _]
images, labels = mnist['data'], mnist['target']

images_train = images[:60000,]
labels_train = labels[:60000,].reshape(1, 60000)



 
encoded_labels = one_hot(labels_train)
 
config = [784, 200, 50, 10]
layers = []

for i in range(len(config) - 1):
    end = False
    if i == len(config) - 2:
        end = True
        
    layer = Layer(config[i], config[i+1], end)
    layer.initial_parameters(None, None)
    layers.append(layer)
    
nn = NeuralNetwork(layers)
train(nn, 0.005, 100, images_train, encoded_labels)

