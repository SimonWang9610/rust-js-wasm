function loadParams(mode) {
	let config_json = window.localStorage.getItem('parameters');
	let config = JSON.parse(config_json);

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
}


function getParameters(mode) {
	var fileUrl = '/static/rust/parameters.json';
	
	if (mode === 'python') {
		fileUrl = '/static/python/parameters.json';
	}
	
	$.ajax({
		type: 'GET',
		cache: false,
		url: fileUrl,
		contentType: 'application/json',
		dataType: 'json'
	}).done((data) => {
		console.log(data);
		// data = JSON.stringify(data);
		window.localStorage.setItem('parameters', JSON.stringify(data));
	});
}

// read the uploaded file as Array
function getImageArray(mode, reader) {

	if (mode === 'rust' && reader.readyState === 2) {
		return new Uint8Array(reader.result)
	} else if (mode === 'python' && reader.readyState === 2) {
		return getBytesInput(reader.result);
	} else {
		return null;
	}
}

function getBytesInput(buffer) {
	var imageBuffer = new Uint8Array(buffer);
	var pixels = [];

	for (let i = 16; i < imageBuffer.length; i++) {
		pixels.push(imageBuffer[i] / 255);
	}
	return new Float32Array(pixels);
}

// set inference mode
function setInferenceMode() {
    let mode = document.getElementById('#inference-mode');
    return mode.options[mode.selectedIndex].text;
}