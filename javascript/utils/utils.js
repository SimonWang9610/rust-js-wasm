const fs = require('fs');
const Layer = require('./layer');
const NeuralNetwork = require('./propagation');
exports.oneHot = (labels) => {
	var newLabels = [];
	labels.forEach((col) => {
		let row = Array(10).fill(0);
		row[col] = 1;
		newLabels.push(row);
	});
	return newLabels;
};

exports.broadcastAdd = (z, bias) => {
	return z.map((row, index) => {
		return row.map((ele) => ele + bias[index]);
	});
};

exports.load = (path) => {
	let rawData = fs.readFileSync(path);
	let parameters = JSON.parse(rawData);

	let network = [];

	Object.values(parameters).forEach((layerJson) => {
		let layer = new Layer(layerJson.neurons, layerJson.prev, layerJson.end);
		layer.initialParameters(layerJson.weights, layerJson.bias);
		network.push(layer);
	});
	return new NeuralNetwork(network);
};
