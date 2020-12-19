const express = require('express');
const bodyParser = require('body-parser');
const methodOverride = require('method-override');
const dot = require('dot');
const fs = require('fs');
const multer = require('multer');
const upload = multer({ dest: 'uploads/' });

const utils = require('./utils/Utils');
const {inference} = require('./www/static/rust/interfaces_node');

const app = express();

app.use(methodOverride());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.text());
app.use(express.static(__dirname + '/www'));

app.get('/', async (req, res, next) => {
	let title = 'Thesis Test';
	let filePath = 'www/templates/index-copy.html';
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

app.post('/predict', upload.array('files[]', 5), (req, res, next) => {

	let mode = req.body.mode;
	let paramsUrl = req.body.paramsUrl;
	let isImage = req.body.isImage;
	console.log('isImage', isImage);

	let file = req.files[0];
	console.log(file);
	
	let buffer = fs.readFileSync(file.path);
	let input = utils.getImagesArray(mode, buffer);

	let params = utils.loadParams(paramsUrl, mode);

	let result = inference(params, input, isImage);
	console.log(result);
	res.json({
		result: result
	});
})

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
