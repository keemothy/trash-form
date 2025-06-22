# Basic Flask backend to serve the model and handle image prediction
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load your trained model
model = tf.keras.models.load_model('best_garbage_model.h5')

# List your class names in the same order as your model's output
CLASS_NAMES = [
    'battery_training_set', 'biological_training_set', 'cardboard_training_set',
    'clothes_training_set', 'glass_training_set', 'metal_training_set',
    'paper_training_set', 'plastic_training_set', 'shoes_training_set', 'trash_training_set'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(preds[0])]
    return jsonify({'class': pred_class})

if __name__ == '__main__':
    app.run(debug=True, port=9000)
