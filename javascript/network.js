const Layer = require('./utils/layer');
const { readMnist } = require('./utils/dataset_node');
const { oneHot, computerLoss, evaluate } = require('./utils/utils');
const { forward, backward } = require('./utils/propagation');
const { transpose } = require('./math.min');
const fs = require('fs');

var paths = [
	[ './mnist/train-images.idx3-ubyte', './mnist/train-labels.idx1-ubyte' ],
	[ './mnist/t10k-images.idx3-ubyte', './mnist/t10k-labels.idx1-ubyte' ]
];
var config = [ 784, 200, 50, 10 ];

var network = [];

const data = readMnist(paths[0], 60000); // images[samples, 784], labels[samples,1]
const images = data[0];
const labels = data[1];

//create network
for (let i = 0; i < config.length - 1; i++) {
	let end = false;
	if (i === config.length - 2) end = true;
	let layer = new Layer(config[i], config[i + 1], end);
	layer.initialParameters(null, null);
	network.push(layer);
}

trainNetwork(5, 0.5, network, images, labels);

function trainNetwork(epoch, alpha, network, images, labels) {
	let transposedImages = transpose(images);
	let encodedLabels = oneHot(labels);

	for (let i = 0; i < epoch; i++) {
		console.log('Epoch#' + i + '# Starting');
		let outputs = forward(network, transposedImages);
		let input = transpose(outputs[-1]);

		let loss = computerLoss(input, encodedLabels);
		let accuracy = evaluate(input, labels);
		console.log('Epoch#' + i + '# : Loss: ' + loss + ' Accuracy: ' + accuracy);
		backward(network, encodedLabels, alpha, outputs);
		console.log('Epoch#' + i + '# Ending');
	}
}
