import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt

# ==================== MODEL ARCHITECTURE ====================

def create_isl_model(num_classes=36, input_shape=(128, 128, 3)):
    """
    Create a CNN model for ISL recognition (A-Z: 26 + 0-9: 10 = 36 classes)
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


# ==================== DATA PREPARATION ====================

class ISLDataPreparator:
    """
    Prepares ISL dataset for training
    Assumes folder structure: dataset/train/{A-Z, 0-9}/ and dataset/test/{A-Z, 0-9}/
    """
    
    def __init__(self, data_dir, img_size=(128, 128), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
    def create_data_generators(self):
        """Create training and validation data generators with augmentation"""
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,  # Keep False for sign language
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Validation data (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        train_dir = os.path.join(self.data_dir, 'train')
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, val_generator


# ==================== TRAINING ====================

class ISLTrainer:
    """Handles model training with callbacks"""
    
    def __init__(self, model, model_save_path='best_isl_model.h5'):
        self.model = model
        self.model_save_path = model_save_path
        
    def compile_model(self):
        """Compile the model with optimizer and loss"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
    def get_callbacks(self):
        """Define training callbacks"""
        callbacks = [
            ModelCheckpoint(
                self.model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=50):
        """Train the model"""
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        return history
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()


# ==================== SENTENCE TO ISL CONVERTER ====================

class SentenceToISLConverter:
    """Converts sentences to ISL signs"""
    
    def __init__(self, model_path, class_labels, img_dir):
        """
        Args:
            model_path: Path to trained model
            class_labels: Dictionary mapping class indices to labels (e.g., {0: 'A', 1: 'B', ...})
            img_dir: Directory containing ISL sign images for display
        """
        self.model = keras.models.load_model(model_path)
        self.class_labels = class_labels
        self.img_dir = img_dir
        self.img_size = (128, 128)
        
    def preprocess_text(self, sentence):
        """Convert sentence to list of valid characters (A-Z, 0-9)"""
        # Remove spaces and convert to uppercase
        sentence = sentence.upper().replace(' ', '')
        # Keep only alphanumeric characters
        valid_chars = [c for c in sentence if c.isalnum()]
        return valid_chars
    
    def get_sign_image_path(self, char):
        """Get the path to the sign image for a character"""
        # Assuming images are stored as: img_dir/{char}/sample_image.jpg
        char_dir = os.path.join(self.img_dir, char)
        if os.path.exists(char_dir):
            images = [f for f in os.listdir(char_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                return os.path.join(char_dir, images[0])
        return None
    
    def display_signs(self, sentence, save_path=None):
        """Display ISL signs for the sentence"""
        chars = self.preprocess_text(sentence)
        
        if not chars:
            print("No valid characters to display")
            return
        
        # Calculate grid size
        n_chars = len(chars)
        cols = min(5, n_chars)
        rows = (n_chars + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        
        for idx, char in enumerate(chars):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            img_path = self.get_sign_image_path(char)
            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(f"'{char}'", fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f"{char}\n(No image)", 
                       ha='center', va='center', fontsize=20)
            
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n_chars, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f"ISL Signs for: '{sentence}'", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
        
        return chars
    
    def convert_sentence(self, sentence, display=True, save_path=None):
        """
        Main function to convert sentence to ISL
        
        Args:
            sentence: Input sentence
            display: Whether to display the signs
            save_path: Path to save the output image
            
        Returns:
            List of characters in the sentence
        """
        print(f"\nOriginal sentence: {sentence}")
        chars = self.preprocess_text(sentence)
        print(f"Characters to sign: {' '.join(chars)}")
        
        if display:
            self.display_signs(sentence, save_path)
        
        return chars


# ==================== MAIN TRAINING SCRIPT ====================

def train_isl_model(data_dir, model_save_path='best_isl_model.h5', epochs=50):
    """
    Main training function
    
    Args:
        data_dir: Root directory containing train/ folder with subdirectories for each class
        model_save_path: Path to save the best model
        epochs: Number of training epochs
    """
    print("=" * 60)
    print("INDIAN SIGN LANGUAGE RECOGNITION - TRAINING")
    print("=" * 60)
    
    # Prepare data
    print("\n[1/4] Preparing data...")
    data_prep = ISLDataPreparator(data_dir, img_size=(128, 128), batch_size=32)
    train_gen, val_gen = data_prep.create_data_generators()
    
    num_classes = len(train_gen.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class labels: {train_gen.class_indices}")
    
    # Create model
    print("\n[2/4] Creating model...")
    model = create_isl_model(num_classes=num_classes)
    model.summary()
    
    # Train model
    print("\n[3/4] Training model...")
    trainer = ISLTrainer(model, model_save_path)
    trainer.compile_model()
    history = trainer.train(train_gen, val_gen, epochs=epochs)
    
    # Plot results
    print("\n[4/4] Plotting training history...")
    trainer.plot_training_history(history)
    
    print(f"\n✓ Training complete! Best model saved to: {model_save_path}")
    print("=" * 60)
    
    return model, history, train_gen.class_indices


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # ===== TRAINING PHASE =====
    # Uncomment to train the model
    """
    DATA_DIR = "path/to/your/dataset"  # Should contain train/{A-Z, 0-9}/ folders
    model, history, class_labels = train_isl_model(
        data_dir=DATA_DIR,
        model_save_path='best_isl_model.h5',
        epochs=50
    )
    """
    
    # ===== INFERENCE PHASE =====
    # After training, use the converter
    """
    # Create class labels dictionary (maps index to character)
    class_labels = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 
        18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P',
        26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
        34: 'Y', 35: 'Z'
    }
    
    # Initialize converter
    converter = SentenceToISLConverter(
        model_path='best_isl_model.h5',
        class_labels=class_labels,
        img_dir='path/to/your/dataset/train'  # Directory with sign images
    )
    
    # Convert sentences to ISL
    converter.convert_sentence("HELLO WORLD", display=True, save_path='hello_world_isl.png')
    converter.convert_sentence("I AM 25 YEARS OLD", display=True)
    converter.convert_sentence("THANK YOU", display=True)
    """
    
    print("\n" + "="*60)
    print("ISL SYSTEM - USAGE INSTRUCTIONS")
    print("="*60)
    print("\n1. TRAINING:")
    print("   - Organize your dataset: dataset/train/{A-Z,0-9}/images.jpg")
    print("   - Uncomment the training section and set DATA_DIR")
    print("   - Run: python script.py")
    print("\n2. INFERENCE:")
    print("   - After training, uncomment the inference section")
    print("   - Use converter.convert_sentence('YOUR TEXT') to display signs")
    print("\n3. DATASET STRUCTURE:")
    print("   dataset/")
    print("   └── train/")
    print("       ├── A/")
    print("       │   ├── img1.jpg")
    print("       │   └── img2.jpg")
    print("       ├── B/")
    print("       ├── ...")
    print("       └── 9/")
    print("="*60)