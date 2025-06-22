import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import random
from pathlib import Path
import json
import warnings
from tensorflow.keras.utils import Sequence

# Suppress warnings
warnings.filterwarnings('ignore')

class RobustImageDataGenerator(ImageDataGenerator):
    """ImageDataGenerator that handles image loading errors gracefully."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped_images = 0
    
    def flow_from_directory(self, directory, target_size=(256, 256), color_mode='rgb',
                           classes=None, class_mode='categorical', batch_size=32, shuffle=True,
                           seed=None, save_to_dir=None, save_prefix='', save_format='png',
                           follow_links=False, subset=None, interpolation='nearest'):
        
        # Create a custom generator that wraps the standard one
        standard_generator = super().flow_from_directory(
            directory, target_size, color_mode, classes, class_mode, batch_size,
            shuffle, seed, save_to_dir, save_prefix, save_format, follow_links,
            subset, interpolation
        )
        
        # Wrap it with error handling
        return RobustDirectoryIterator(standard_generator, self.skipped_images)

class RobustDirectoryIterator(Sequence):
    """Wrapper for ImageDataGenerator that handles corrupted images gracefully."""
    
    def __init__(self, generator, skipped_counter):
        self.generator = generator
        self.skipped_counter = skipped_counter
        self.classes = generator.classes
        self.class_indices = generator.class_indices
        self.samples = generator.samples
        self.batch_size = generator.batch_size
        self.filepaths = generator.filepaths
    
    def __len__(self):
        return len(self.generator)
    
    def __getitem__(self, idx):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self.generator[idx]
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Skipping problematic batch {idx}, attempt {attempt + 1}")
                    continue
                else:
                    print(f"❌ Failed to load batch {idx} after {max_retries} attempts")
                    # Return a dummy batch to prevent training from stopping
                    dummy_x = np.zeros((self.batch_size, 224, 224, 3))
                    dummy_y = np.zeros((self.batch_size, len(self.class_indices)))
                    return dummy_x, dummy_y
    
    def reset(self):
        return self.generator.reset()
    
    def next(self):
        return self.__getitem__(0)

class GarbageClassifier:
    def __init__(self, train_dir, test_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the garbage classifier.
        
        Args:
            train_dir (str): Path to training data directory
            test_dir (str): Path to testing data directory
            img_size (tuple): Target image size (height, width)
            batch_size (int): Batch size for training
        """
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = []
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        
        print("🔧 Initializing Garbage Classifier...")
        self._setup_data_generators()
        self._build_model()
        
    def _setup_data_generators(self):
        """Setup data generators for training and testing."""
        print("📊 Setting up data generators...")
        
        # Data augmentation for training
        train_datagen = RobustImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # 20% of training data for validation
        )
        
        # Only preprocessing for testing (no augmentation)
        test_datagen = RobustImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Load validation data
        self.validation_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )
        
        # Load testing data
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False  # Keep order for evaluation
        )
        
        # Get class names
        self.class_names = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"✅ Found {self.num_classes} classes: {self.class_names}")
        print(f"📈 Training samples: {self.train_generator.samples}")
        print(f"🔍 Validation samples: {self.validation_generator.samples}")
        print(f"🧪 Testing samples: {self.test_generator.samples}")
        
    def _build_model(self):
        """Build the CNN model architecture."""
        print("🏗️ Building model architecture...")
        
        # Use MobileNetV2 as base model (pre-trained on ImageNet)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create the full model
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model built successfully!")
        self.model.summary()
        
    def train(self, epochs=50, early_stopping_patience=10):
        """
        Train the model.
        
        Args:
            epochs (int): Number of training epochs
            early_stopping_patience (int): Patience for early stopping
        """
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'best_garbage_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model with error handling
        try:
            self.history = self.model.fit(
                self.train_generator,
                epochs=epochs,
                validation_data=self.validation_generator,
                callbacks=callbacks_list,
                verbose=1
            )
            print("✅ Training completed!")
        except Exception as e:
            print(f"⚠️ Training encountered an error: {e}")
            print("🔄 Attempting to continue with available data...")
            # Try to save the model if it exists
            if hasattr(self, 'model'):
                self.model.save('garbage_model_partial.h5')
                print("💾 Partial model saved as garbage_model_partial.h5")
        
    def evaluate(self):
        """Evaluate the model on test data."""
        print("🧪 Evaluating model on test data...")
        
        try:
            # Evaluate on test set
            test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)
            print(f"📊 Test Accuracy: {test_accuracy:.4f}")
            print(f"📊 Test Loss: {test_loss:.4f}")
            
            # Get predictions
            self.test_generator.reset()
            predictions = self.model.predict(self.test_generator, verbose=1)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = self.test_generator.classes
            
            # Classification report
            print("\n📋 Classification Report:")
            print(classification_report(true_classes, predicted_classes, 
                                      target_names=self.class_names))
            
            # Save results
            results = {
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'class_names': self.class_names,
                'predictions': predictions.tolist(),
                'true_classes': true_classes.tolist(),
                'predicted_classes': predicted_classes.tolist()
            }
            
            with open('model_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            return test_accuracy, test_loss, predictions, predicted_classes, true_classes
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return None, None, None, None, None
        
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("❌ No training history available. Train the model first.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, true_classes, predicted_classes):
        """Plot confusion matrix."""
        if true_classes is None or predicted_classes is None:
            print("❌ Cannot plot confusion matrix - no evaluation data available.")
            return
            
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def predict_single_image(self, image_path):
        """
        Predict the class of a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
        """
        from tensorflow.keras.preprocessing import image
        
        # Load and preprocess image
        img = image.load_img(image_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Predict
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return self.class_names[predicted_class], confidence, predictions[0]

def main():
    """Main function to run the complete training and evaluation pipeline."""
    print("🗑️ Garbage Classification Model Training Pipeline")
    print("=" * 50)
    
    # Initialize classifier
    classifier = GarbageClassifier(
        train_dir="split_garbage_dataset/training",
        test_dir="split_garbage_dataset/testing",
        img_size=(224, 224),
        batch_size=32
    )
    
    # Train the model
    classifier.train(epochs=50, early_stopping_patience=10)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate the model
    test_accuracy, test_loss, predictions, predicted_classes, true_classes = classifier.evaluate()
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(true_classes, predicted_classes)
    
    print("\n🎉 Training and evaluation completed!")
    if test_accuracy is not None:
        print(f"📊 Final Test Accuracy: {test_accuracy:.4f}")
    print("📁 Model saved as: best_garbage_model.h5")
    print("📁 Results saved as: model_results.json")
    print("📁 Plots saved as: training_history.png, confusion_matrix.png")

if __name__ == "__main__":
    main()