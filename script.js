// Variables globales
const modeSelect = document.getElementById('mode-select');
const uploadInput = document.getElementById('upload');
const previewImg = document.getElementById('preview');
const videoPreview = document.getElementById('video-preview');
const captureBtn = document.getElementById('capture');
const resultDiv = document.getElementById('prediction-text');

let model; // Para almacenar el modelo de TensorFlow.js

// Cargar el modelo de TensorFlow.js
async function loadModel() {
    console.log("Cargando modelo...");
    model = await tf.loadLayersModel('ruta/a/tu/modelo/model.json'); // Cambia la ruta a tu modelo
    console.log("Modelo cargado.");
}

// Preprocesar la imagen para el modelo
function preprocessImage(imageElement) {
    return tf.tidy(() => {
        // Convertir la imagen a un tensor
        const tensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224]) // Redimensionar a 224x224 (tamaño esperado por VGG16)
            .toFloat();

        // Normalizar la imagen (si es necesario)
        const normalized = tensor.div(255.0); // Normalizar a [0, 1]
        const batched = normalized.expandDims(0); // Añadir dimensión del batch
        return batched;
    });
}

// Realizar la predicción
async function predict(imageElement) {
    if (!model) {
        console.error("El modelo no está cargado.");
        return;
    }

    // Preprocesar la imagen
    const tensor = preprocessImage(imageElement);

    // Realizar la predicción
    const predictions = await model.predict(tensor);
    const predictedClass = predictions.argMax(1).dataSync()[0]; // Obtener la clase predicha
    const confidence = predictions.max().dataSync()[0]; // Obtener la confianza

    // Mostrar el resultado
    resultDiv.textContent = `Predicción: ${predictedClass} (Confianza: ${confidence.toFixed(2)})`;
}

// Manejar el cambio en el combobox
modeSelect.addEventListener('change', function (event) {
    const selectedMode = event.target.value;

    // Ocultar todos los contenedores
    document.getElementById('upload-container').style.display = 'none';
    document.getElementById('camera-container').style.display = 'none';
    previewImg.style.display = 'none';

    // Mostrar el contenedor correspondiente
    if (selectedMode === 'upload') {
        document.getElementById('upload-container').style.display = 'block';
    } else if (selectedMode === 'camera') {
        document.getElementById('camera-container').style.display = 'block';
        startCamera();
    }
});

// Manejar la subida de una imagen
uploadInput.addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            predict(previewImg); // Realizar la predicción
        };
        reader.readAsDataURL(file);
    }
});

// Iniciar la cámara
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoPreview.srcObject = stream;
    } catch (error) {
        console.error("Error al acceder a la cámara:", error);
    }
}

// Capturar imagen desde la cámara
captureBtn.addEventListener('click', function () {
    const canvas = document.createElement('canvas');
    canvas.width = videoPreview.videoWidth;
    canvas.height = videoPreview.videoHeight;
    canvas.getContext('2d').drawImage(videoPreview, 0, 0, canvas.width, canvas.height);
    previewImg.src = canvas.toDataURL();
    previewImg.style.display = 'block';
    predict(previewImg); // Realizar la predicción
});

// Cargar el modelo al iniciar la aplicación
loadModel();