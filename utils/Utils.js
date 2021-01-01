const fs = require('fs');

let utils = {
	loadParams: (fileUrl, mode) => {
		let rawData = fs.readFileSync(fileUrl);
		let config = JSON.parse(rawData);
	
		if (mode === 'python') {
			for (let i = 0; i < 3; i++) {
				config[i].weights = new Float32Array(config[i].weights.flat());
				config[i].bias = new Float32Array(config[i].bias.flat());
			}
			return config
		} else if (mode === 'rust') {
			let params = [];
			for (let key of Object.keys(config)) {
				params.push(config[key]);
			}
			return params; // [{Object}]
		}
	},

	getImagesArray: (mode, buffer) => {
		if (mode === 'python') {
			return getBytesInput(buffer);
		} else if (mode === 'rust') {
			return new Uint8Array(buffer);
		}
	}
};

function getBytesInput(buffer) {
	var imageBuffer = new Uint8Array(buffer);
	var pixels = [];

	for (let i = 16; i < imageBuffer.length; i++) {
		pixels.push(imageBuffer[i] / 255);
	}
	return new Float32Array(pixels);
}

module.exports = utils;