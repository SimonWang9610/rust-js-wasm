$(document).ready(async () => {
	getParameters();
});

var reader = new FileReader();
var isImage = false;
document.querySelector('#upload').addEventListener('change', function(e) {
	if (this.files[0].type === 'image/png') {
		isImage = true;
	}
	reader.readAsArrayBuffer(this.files[0]);
});

function getImageArray() {
	if (reader.readyState === 2) {
		return new Uint8Array(reader.result);
	} else {
		return null;
	}
}

function getParameters() {
	$.ajax({
		type: 'GET',
		cache: false,
		url: '/static/parameters.json',
		contentType: 'application/json',
		dataType: 'json'
	}).done((data) => {
		console.log(data);
		// data = JSON.stringify(data);
		window.localStorage.setItem('parameters', JSON.stringify(data));
	});
}
