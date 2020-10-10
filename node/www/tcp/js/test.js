import init, { inference } from '/static/wasm_game_of_life.js';

async function initNetwork() {
	let params = getParams();
	let buffer = reader.result;
	console.log('initNetwork -> buffer', buffer);

	await init();

	let result = inference(params, buffer);
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
