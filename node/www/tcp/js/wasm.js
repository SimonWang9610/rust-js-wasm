import init, { inference } from '/static/interfaces.js';
// import config from '/static/parameters.js';

async function initNetwork() {
	// let images = getImages();
	let params = getParams();

	let imageArray = getImageArray();
	console.log('isImage ' + isImage);
	await init();

	let result = inference(params, imageArray, isImage);
	console.log('initNetwork -> result', result);
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

$('#layers').bind('click', function(e) {
	e.preventDefault();
	e.stopPropagation();
	initNetwork();
});
