const math = require('./math.min');
const { computeLoss, oneHot, evaluate } = require('./utils/utils');
const { readMnist } = require('./utils/dataset_node');
const Layer = require('./utils/layer');
const NeuralNetwork = require('./utils/propagation');
const { forward, backward } = require('./utils/propagation');

var paths = [
	[ './mnist/train-images.idx3-ubyte', './mnist/train-labels.idx1-ubyte' ],
	[ './mnist/t10k-images.idx3-ubyte', './mnist/t10k-labels.idx1-ubyte' ]
];
var config = [ 784, 200, 50, 10 ];

let data = readMnist(paths[0], 60000);
let images = data[0];
let labels = data[1];
let encodedLabels = oneHot(labels);

let network = [];

for (let i = 0; i < config.length - 1; i++) {
	let end = false;
	if (i === config.length - 2) end = true;
	let layer = new Layer(config[i], config[i + 1], end);
	layer.initialParameters(null, null);
	network.push(layer);
}

let nn = new NeuralNetwork(network);
console.log('nn', nn);

let imagesT = math.transpose(images);
console.log('images', math.size(imagesT));

console.log('computing');

nn.forward(imagesT);
let loss = nn.computeLoss(encodedLabels);
console.log('loss', loss);

let accuracy = nn.evaluate(labels);
console.log('accuracy', accuracy);

console.log('Starting back propagation....');
nn.backward(0.5, encodedLabels);

console.log('Ending...');

// let a = [ 1, 2, 3, 4, 5 ];
// let outputs = [ 1 ];

// a.reduce((acc, val) => {
// 	let temp = acc + 1;
// 	outputs.push(temp);
// 	return temp;
// }, outputs[0]);

// console.log(outputs);
