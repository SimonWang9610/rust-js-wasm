const math = require('../math.min');
const { relu, softmax } = require('./activation');
const { broadcastAdd } = require('./utils');

class Layer {
	constructor(...args) {
		this.neurons = args[1];
		this.prev = args[0];
		this.end = args[2];
		this.weights = null;
		this.bias = null;
	}

	initialParameters(weights, bias) {
		if (weights && bias) {
			this.weights = weights;
			this.bias = bias;
		} else {
			this.weights = math.multiply(math.random([ this.neurons, this.prev ], -1, 1), 0.01);
			this.bias = Array(this.neurons).fill(0);
		}
	}

	// wx + b
	forward(input) {
		if (this.weights === null) {
			throw new Error('no initialization of weights');
		}

		let z = broadcastAdd(math.multiply(this.weights, input), this.bias);

		if (this.end) {
			return softmax(z);
		} else {
			return relu(z);
		}
	}
}

module.exports = Layer;
