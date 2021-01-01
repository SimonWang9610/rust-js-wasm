var mode = setInferenceMode();
var input = null;

var reader = new FileReader();
var isImage = false;
document.querySelector('#upload').addEventListener('change', function(e) {
	if (this.files[0].type === 'image/png') {
		isImage = true;
	}
	reader.readAsArrayBuffer(this.files[0]);
});

$('#inference-mode').on('change', () => {
    mode = setInferenceMode();

    if (mode === 'client-python') {
        requestPythonFiles();
    } else if (mode === 'client-rust') {
        requestRustFiles();
    }

});


