const fs = require('fs');
const utils = require('./utils/Utils');

let url = './uploads/34fbdd2a7b78de99e159a19f9e0a668f';

let buffer = fs.readFileSync(url);
let images = utils.getImagesArray('rust', buffer);
console.log(images.length);