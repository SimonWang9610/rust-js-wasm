function getImageArray(reader) {
	if (reader.readyState === 2) {
		return new Uint8Array(reader.result);
	} else {
		return null;
	}
}

function read(file) {
	var reader = new FileReader();
	reader.readAsArrayBuffer(file);

	return reader;
}

function getParams() {
	let config_json = window.localStorage.getItem('parameters');
	let config = JSON.parse(config_json);

	let params = [];
	for (let key of Object.keys(config)) {
		params.push(config[key]);
	}
	return params; // [{Object}]
}

function getParameters() {
	$.ajax({
		type: 'GET',
		cache: false,
		url: '/static/parameters-32-6.json',
		contentType: 'application/json',
		dataType: 'json'
	}).done((data) => {
		console.log("parameters loaded: " + data);
		createDB(data);
		// data = JSON.stringify(data);
		window.localStorage.clear();
		window.localStorage.setItem('parameters', JSON.stringify(data));
	});
}