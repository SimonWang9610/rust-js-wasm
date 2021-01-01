// var predictRust = import('/static/rust/wasm.js');

////////////////////////////// download and load parameters.json
function loadParams(mode) {
	let config_json = window.localStorage.getItem(mode + 'Params');
	console.assert(config_json != null);
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

function downloadParameters(mode) {
	var fileUrl = '/static/' + mode + '/parameters.json';
	var fileName = mode + 'Params';
	$.ajax({
		type: 'GET',
		cache: false,
		url: fileUrl,
		contentType: 'application/json',
		dataType: 'json'
	}).done((data) => {
		console.log(data);
		// data = JSON.stringify(data);
		window.localStorage.clear();
		window.localStorage.setItem(fileName, JSON.stringify(data));
		requestFiles(mode);
	})
}


//////////////////////////////////////////// read the uploaded file as Array
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


////////////////////////////////////// set inference mode
function setInferenceMode() {
    let mode = document.getElementById('inference-mode');
    return mode.options[mode.selectedIndex].text;
}

//////////////////////////////////////////// request required files
// rust: /static/rust/wasm.js
// python: /static/python/pyodide.js,  /static/python/python.js
function requestFiles(mode) {
	var body = $('body');

	if (mode === 'rust') {
		let script = createScriptTag('/static/rust/wasm.js', 'rust');
		if (!loadedRust) body.append(script);
		loadedRust = true;
	} else if (mode === 'python') {
		let pyodide = createScriptTag('/static/python/pyodide.js');
		let python = createScriptTag('/static/python/python.js');
		
		if (!loadedPython) {
			$('head').append(pyodide);
			body.append(python);
		}

		loadedPython = true;
	}

}

function createScriptTag(url, mode) {
	var script = document.createElement('script');
	if (mode === 'rust') {
		script.type = 'module';
	} else {
		script.type = 'text/javascript';

	}
	script.src = url;
	return script;
}
