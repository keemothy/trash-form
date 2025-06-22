import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

def predict_image(model_path, image_path, class_names):
    """
    Predict the class of a single image.
    
    Args:
        model_path (str): Path to the trained model
        image_path (str): Path to the image to predict
        class_names (list): List of class names
        
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    # Load the model
    model = load_model(model_path)
    
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]

def display_prediction(image_path, predicted_class, confidence, all_probabilities, class_names):
    """Display the image with prediction results."""
    # Load image for display
    img = image.load_img(image_path, target_size=(224, 224))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    ax1.imshow(img)
    ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2%}')
    ax1.axis('off')
    
    # Display probability distribution
    bars = ax2.bar(range(len(class_names)), all_probabilities)
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Probability')
    ax2.set_title('Class Probabilities')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    
    # Highlight the predicted class
    bars[predicted_class_idx].set_color('red')
    
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py <image_path>")
        print("Example: python predict_image.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    model_path = "best_garbage_model.h5"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train the model first using garbage_classifier.py")
        sys.exit(1)
    
    # Class names (should match the order from training)
    class_names = [
        'battery_training_set',
        'biological_training_set', 
        'cardboard_training_set',
        'clothes_training_set',
        'glass_training_set',
        'metal_training_set',
        'paper_training_set',
        'plastic_training_set',
        'shoes_training_set',
        'trash_training_set'
    ]
    
    print(f"🔍 Predicting class for: {image_path}")
    
    try:
        predicted_class, confidence, all_probabilities = predict_image(
            model_path, image_path, class_names
        )
        
        # Clean up class name for display
        display_class = predicted_class.replace('_training_set', '').title()
        
        print(f"✅ Prediction: {display_class}")
        print(f"📊 Confidence: {confidence:.2%}")
        print(f"📈 All probabilities:")
        
        for i, (class_name, prob) in enumerate(zip(class_names, all_probabilities)):
            clean_name = class_name.replace('_training_set', '').title()
            print(f"   {clean_name}: {prob:.2%}")
        
        # Display the image with prediction
        display_prediction(image_path, display_class, confidence, all_probabilities, class_names)
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 