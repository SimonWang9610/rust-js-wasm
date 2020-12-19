// 直接读取 parameters.json 中的 weights 和 bias 会导致在 initial_parameters时出现问题，报错 cannot operand JsProxy with float
// Solution：把 weights&bias 展开为一维数组，并且转换成 TypedArray：Float32Array， 然后在 python 中 reshape

// reshape 输入时一定要是 （10000， 784）

// *!!!!!!!最后预测结果全是 7. 一定哪里出了问题
// 如果不考虑准确度的话，需要排除一个影响因素：参数的数量级的不同是否会明显改变 weight * input + bias 的计算时间
// 猜想原因：
//          1） 如果数量级的差别对计算时间没有明显影响的话，那我的测试就可以忽略模型准确度的问题了，主要保证网路结构和输入数据一致，然后进行计算时间的对比
//          2） 同时，也使得 rust 模型的运算时间具有可比较性，因为 python 模型的参数和 rust 模型的参数在数量级上有明显的差异
// 可参考的研究方向： 浮点数的乘法和加法运算

getParameters();
var parameters = getParams();
var input = null;

function predictPython(reader) {
    input = getImageArray(reader);
    runPyodide();
}

function runPyodide() {
    languagePluginLoader.then(() => {
        pyodide.loadPackage('numpy').then(() => { 
            pyodide.runPython(`
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
            images = np.asarray(input, dtype=np.float32).reshape((10000, 784))
            print(type(images))
            print(nn.predict(images))
            `);
        }) 
    });
}

