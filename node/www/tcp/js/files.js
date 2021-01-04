var isImage = false;
var db;
var imageReader = null;
var labelReader = null;

$(document).ready(async () => {
	// getParameters();
	// fetchParamFile();
	// let data = await fetchParamFile('/static/parameters-32-q.json');
	let data = null;
	createDB(data, 4, "layer6");
});

document.querySelector("#upload").addEventListener("change", function (e) {
	
	if (this.files[0].type === 'image/png') {
		isImage = true;
	}

	imageReader = read(this.files[0]);
});

document.querySelector("#labels").addEventListener("change", function (e) {
	labelReader = read(this.files[0]);
});


function createDB(objs, version, name) {
	let openRequest = indexedDB.open("db", version);
	
	openRequest.onsuccess = function () {
		console.log("Database opened!");
		db = openRequest.result;
	};

	openRequest.onerror = function () {
		console.log("Failed to open database");	
	};

	openRequest.onupgradeneeded = function (e) {
		let db = openRequest.result;
		
		if (!db.objectStoreNames.contains(name)) {
			let objStore = db.createObjectStore(name, { autoIncrement: true });

			objStore.createIndex("neurons", "neurons", {unique: false});
			objStore.createIndex("prev", "prev", {unique: false});
			objStore.createIndex("weights", 'weights', {unique: false});
			objStore.createIndex("bias", "bias", {unique: false});
			objStore.createIndex("end", "end", { unique: false });
			objStore.createIndex("factor", "factor", { unique: false });
			objStore.createIndex("zero", "zero", { unique: false });
			
			objStore.transaction.oncomplete = function (e) {
				let layerObjStore = db.transaction(name, "readwrite").objectStore(name);
				for (let key of Object.keys(objs)) {
					console.log("adding [" + key + "] to DB: " + objs[key]);
					layerObjStore.add(objs[key]);
				}
			}
			console.log("Database setup completely!");
		}
	}

}

async function fetchParamFile(path) {
	let file = await fetch(path)
		.then((res) => {
			if (!res.ok) {
				throw new Error("HTTP Error! Status: " + res.status);
			}
			return res.blob();
		});
	
	let str = await file.text();
	return JSON.parse(str);
}

// document.querySelector('#upload').addEventListener('change', function(e) {
// 	if (this.files[0].type === 'image/png') {
// 		isImage = true;
// 	}
// 	reader.readAsArrayBuffer(this.files[0]);
// });