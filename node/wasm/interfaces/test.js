const fs = require('fs');
const buf = fs.readFileSync('./pkg/interfaces_bg.wasm');
const params = fs.readFile('./pkg/parameters.json', (err, data) => {
	if (err) throw err;
	return JSON.parse(data);
});

console.log('params: ' + params);

const instance = async () => {
	return await WebAssembly.instantiate(new Uint8Array(buf)).then((res) => res.instantiate.exports);
};
