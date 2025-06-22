// Basic frontend for uploading an image and sending it to the backend for prediction

document.getElementById('imageInput').addEventListener('change', function (e) {
    const file = e.target.files[0];
    const preview = document.getElementById('preview');
    if (file) {
        const reader = new FileReader();
        reader.onload = function (evt) {
            preview.src = evt.target.result;
            preview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    } else {
        preview.src = '#';
        preview.style.display = 'none';
    }
});

document.getElementById('predictBtn').addEventListener('click', async function () {
    const input = document.getElementById('imageInput');
    const resultDiv = document.getElementById('result');
    if (!input.files.length) {
        resultDiv.textContent = 'Please select an image.';
        return;
    }
    const file = input.files[0];
    const formData = new FormData();
    formData.append('image', file);

    resultDiv.textContent = 'Classifying...';
    try {
        const response = await fetch('http://127.0.0.1:8899/predict', {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Prediction failed');
        const data = await response.json();
        resultDiv.textContent = 'Prediction: ' + data.class;
    } catch (err) {
        resultDiv.textContent = 'Error: ' + err.message;
    }
});
