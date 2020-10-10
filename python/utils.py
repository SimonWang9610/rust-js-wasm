import numpy as np

def relu(x):
    return (x >= 0).astype(int) * x

def softmax(x):
    temp = x - x.max()
    return np.exp(temp) / np.sum(np.exp(temp), axis=0, keepdims=True)

def relu_derivate(x):
    return (x >= 0).astype(int)

def one_hot(labels):
    temp = np.zeros((10, labels.shape[1]))
    
    for i in range(labels.shape[1]):
        temp[int(labels[0, i]), i] = 1
    return temp
    
    
    
    
