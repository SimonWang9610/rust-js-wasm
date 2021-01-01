$(document).ready(() => {
	getParameters();
});

// var reader = new FileReader();
var isImage = false;

var imageReader = null;
var labelReader = null;

// document.querySelector('#upload').addEventListener('change', function(e) {
// 	if (this.files[0].type === 'image/png') {
// 		isImage = true;
// 	}
// 	reader.readAsArrayBuffer(this.files[0]);
// });

document.querySelector("#upload").addEventListener("change", function (e) {
	
	if (this.files[0].type === 'image/png') {
		isImage = true;
	}

	imageReader = read(this.files[0]);
});

document.querySelector("#labels").addEventListener("change", function (e) {
	labelReader = read(this.files[0]);
});

function getImageArray(reader) {
	if (reader.readyState === 2) {
		return new Uint8Array(reader.result);
	} else {
		return null;
	}
}

// function getLabelArray(reader) {
// 	if (reader.readyState === 2) {
// 		return new Uint8Array(reader.result);
// 	} else {
// 		return null;
// 	}
// }

function read(file) {
	var reader = new FileReader();
	reader.readAsArrayBuffer(file);

	return reader;
}

function getParameters() {
	$.ajax({
		type: 'GET',
		cache: false,
		url: '/static/parameters-32.json',
		contentType: 'application/json',
		dataType: 'json'
	}).done((data) => {
		console.log(data);
		// data = JSON.stringify(data);
		window.localStorage.clear();
		window.localStorage.setItem('parameters', JSON.stringify(data));
	});
}
