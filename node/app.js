const express = require('express');
const bodyParser = require('body-parser');
const cookieParser = require('cookie-parser');
const methodOverride = require('method-override');
const session = require('express-session');
const dot = require('dot');
const fs = require('fs');

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
