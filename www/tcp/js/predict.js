
var mode = setInferenceMode();
var input = null;
var loadedRust = true;
var loadedPython = false;

var reader = new FileReader();
var isImage = false;
document.querySelector('#upload').addEventListener('change', function(e) {
	if (this.files[0].type === 'image/png') {
		isImage = true;
	}
	reader.readAsArrayBuffer(this.files[0]);
});


function predictServer() {
    let backend = 'rust';

    if (mode === 'server-python') {
        backend = 'python';
    }

    let paramsUrl = 'www/static/' + backend + '/parameters.json';
    let file = document.getElementById('upload').files[0];

    if (file.type === 'image/png') {
        isImage = true;
    } else {
        isImage = false;
    }
    uploadImage(file, backend, paramsUrl, isImage);

}

function uploadImage(file, backend, url, isImage) {
    let formData = new FormData();
    formData.append('files[]', file);
    formData.append('mode', backend);
    formData.append('paramsUrl', url);
    formData.append('isImage', isImage);
    console.log(formData);
    $.ajax({
        url: '/predict',
        type: 'POST',
        data: formData,
        async: false,
        contentType: false,
        enctype: 'multipart/form-data',
        processData: false,
        success: function (data) {
            console.log('predictions: ', data.result);
        },
        error: function (xhr) { }
    })
}
