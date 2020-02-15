var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
  };
  reader.readAsDataURL(input.files[0]);
}

function analyze() {
  var uploadFiles = el("file-input").files;
  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

  el("analyze-button").innerHTML = "Analyzing...";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
    true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };

  xhr.onload = function(e) {
    if (this.readyState === 4) {
      var binaryData = [];
      binaryData.push(e.target.response);
      const blobUrl = URL.createObjectURL(new Blob(binaryData, {type: "image/png"}))
      //const blobUrl = URL.createObjectURL(e.target.response);
      //el("image-picked").src = 'data:image/png;base64,'+b64Response;
  el("image-result").src = blobUrl;
    }
    el("analyze-button").innerHTML = "Analyze";
    //el("mel-button").innerHTML = "Generate Mel Spectrogram"
  };

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}

