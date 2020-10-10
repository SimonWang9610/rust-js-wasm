const math = require('../math.min');
const { reluDerivate } = require('./activation');
const { broadcastAdd } = require('./utils');
const fs = require('fs');

function optimize(layer, DZ, DW, alpha, input) {
	layer.weights = layer.weights - math.divide(math.multiply(alpha, DW), input[0].length);
	layer.bias = layer.bias - math.divide(math.multiply(alpha, DZ.map((row) => math.sum(row))), input.length);
}

class NeuralNetwork {
	constructor(layers) {
		this.layers = layers;
		this.outputs = null;
	}

	forward(input) {
		let outputs = [ input ];

		this.layers.reduce((acc, layer) => {
			let output = layer.forward(acc);
			outputs.push(output);
			return output;
		}, outputs[0]);

		this.outputs = outputs;
	}

	backward(alpha, target) {
		let weight = math.transpose(this.layers[this.layers.length - 1].weights);
		let DZ = null;
		let DW = null;

		for (let layer of this.layers.reverse()) {
			let output = this.outputs.pop();
			let input = this.outputs[this.outputs.length - 1];

			if (layer.end) {
				let sample = target.length;
				DZ = math.divide(math.subtract(output - math.transpose(target)), sample);
			} else {
				let x = broadcastAdd(math.multiply(layer.weights, input), layer.bias);
				DZ = math.dotMultiply(math.multiply(weight, DZ), reluDerivate(x));
				weight = math.transpose(layer.weights);
			}

			let transposedInput = math.transpose(input);
			DW = math.multiply(DZ, transposedInput);
			optimize(layer, DZ, DW, alpha, transposedInput);
		}
	}

	evaluate(target) {
		let output = this.outputs[this.outputs.length - 1];
		let total = target.length;
		let predictions = output.map((row) => {
			let max = row[0];
			return row.reduce((acc, val, index) => {
				if (val > max) {
					max = val;
					return index;
				} else {
					return acc;
				}
			}, 0);
		});

		let correct = predictions.reduce((acc, val, index) => {
			if (val === target[index]) {
				return acc + 1;
			} else {
				return acc;
			}
		}, 0);

		return correct / total;
	}

	computeLoss(labels) {
		let output = this.outputs[this.outputs.length - 1];
		let sample = labels.length;
		let y = math.flatten(output);
		let t = math.flatten(labels);

		return (
			-(1 / sample) *
			y.reduce((acc, val, index) => {
				return acc + t[index] * math.log(val);
			}, 0)
		);
	}

	save() {
		let parameters = {};

		for (let i = 0; i < this.layers.length; i++) {
			let layerJson = {
				neurons: this.layers[i].neurons,
				prev: this.layers[i].prev,
				weights: this.layers[i].weights,
				bias: this.layers[i].bias,
				end: this.layers[i].end
			};
			parameters.push({ i: layerJson });
		}
		fs.writeFile('parameters.json', JSON.stringify(parameters));
	}

	predict(input) {
		let output = this.layers.reduce((acc, layer) => {
			return layer.forward(acc);
		}, input);

		return output.map((row) => {
			let max = row[0];
			return row.reduce((acc, val, index) => {
				if (val > max) {
					max = val;
					return index;
				} else {
					return acc;
				}
			}, 0);
		});
	}
}

module.exports = NeuralNetwork;
