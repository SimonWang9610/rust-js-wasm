import init, { inference } from '/static/interfaces.js';
// import config from '/static/parameters.js';

async function initNetwork(params, quantized) {
	// let images = getImages();
    console.log('🚀 ~ file: wasm.js ~ line 7 ~ initNetwork ~ params', params)
	let imageArray = getImageArray(imageReader);
    console.log('🚀 ~ file: wasm.js ~ line 9 ~ initNetwork ~ imageArray', imageArray)

	let labels = getImageArray(labelReader);
    console.log('🚀 ~ file: wasm.js ~ line 12 ~ initNetwork ~ labels', labels)
	
	await init();

	let start = Date.now();
	let result = inference(params, imageArray, labels, isImage, quantized);
	let end = Date.now() - start;

	console.log("total time: " + end);
	console.log("result " + result);
}


async function paramsFromDB(name, depth) {
	let layerObjStore = db.transaction(name, "readonly").objectStore(name);
	let params = [];

	for (let i = 1; i < depth; i++) {
		await getById(layerObjStore, i).then(layer => {
			params.push(layer);
		});
		
	}
	return params;

}

function getById(store, id) {
	return new Promise((resolve) => {
		let request = store.get(id);

		request.onsuccess = function (e) {
			return resolve(request.result)
		}
	})
}


$('#layers').bind('click', function(e) {
	e.preventDefault();
	e.stopPropagation();
	// initNetwork();
	paramsFromDB("layer3", 4).then(params => {
		initNetwork(params, false);
	})
});
