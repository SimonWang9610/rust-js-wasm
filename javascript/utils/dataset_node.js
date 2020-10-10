const fs = require('fs');

exports.readMnist = (path, samples) => {
	var imageFileBuffer = fs.readFileSync(path[0]);
	var labelsBuffer = fs.readFileSync(path[1]);
	var images = [];
	var labels = [];

	for (let image = 0; image < samples; image++) {
		var pixels = [];
		for (let x = 0; x < 28; x++) {
			for (let y = 0; y < 28; y++) {
				pixels.push(imageFileBuffer[image * 28 * 28 + (x + y * 28) + 15]);
			}
		}
		images.push(pixels);
		labels.push(labelsBuffer[image + 8]);
	}
	return [ images, labels ];
};
