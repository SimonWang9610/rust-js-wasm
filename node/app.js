const express = require('express');
const bodyParser = require('body-parser');
const methodOverride = require('method-override');
const dot = require('dot');
const fs = require('fs');
const { EEXIST } = require('constants');

const app = express();

app.use(methodOverride());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.text());
app.use(express.static(__dirname + '/www'));

app.get('/', async (req, res, next) => {
	let title = 'Thesis Test';
	let filePath = 'www/templates/index.html';
	let contentType = 'text/html';

	try {
		let data = await readFile(filePath);
		let templateParams = {
			Title: title
		};
		let htmlTemplate = data.toString();
		let templateFn = dot.template(htmlTemplate);
		let html = templateFn(templateParams);
		res.writeHead(200, { 'Content-ype': contentType });
		res.end(html);
	} catch (err) {
		res.end('Internal serve error');
	}
});

app.get('/parameters', async (req, res, next) => {
	fs.readFile('./www/static/parameters-32-6.json', "utf-8", (err, jsonString) => {
		if (err) {
			throw err;
		}

		try {
			let layers = JSON.parse(jsonString);
			console.log("layers: " + layers);
		} catch (error) {
			console.log("Error: " + error);
		}
	});
});

function readFile(path) {
	return new Promise((resolve, reject) => {
		fs.readFile(path, (err, data) => {
			if (err) {
				reject(err);
			} else {
				resolve(data);
			}
		});
	});
}

app.listen(8000, (err) => {
	console.log('Server is listening on port 8000');
});
