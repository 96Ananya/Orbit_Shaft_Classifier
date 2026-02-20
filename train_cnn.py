import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing
import os
import numpy as np

# --- CONFIGURATION ---
DATASET_PATH = "orbit_dataset"
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20
MODEL_FILE = "vibration_model.h5"

# --- DATA LOADING ---
def load_and_combine_data():
    """Loads images from 'clean' variants and combines them."""
    
    print(f"Loading data from {DATASET_PATH}...")
    
    # Check if the dataset directory exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path '{DATASET_PATH}' not found. Please run the data generator script first.")

    # Load data for the 'clean' variant
    clean_dir = os.path.join(DATASET_PATH, 'clean')
    clean_ds = preprocessing.image_dataset_from_directory(
        clean_dir,
        labels='inferred',
        label_mode='categorical', 
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # Load data for the 'realistic' variant
    realistic_dir = os.path.join(DATASET_PATH, 'realistic')
    realistic_ds = preprocessing.image_dataset_from_directory(
        realistic_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Combine the datasets
    combined_ds = clean_ds.concatenate(realistic_ds)
    
    # Determine class names 
    class_names = clean_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Count total number of images (approximate)
    total_images = len(clean_ds) * BATCH_SIZE + len(realistic_ds) * BATCH_SIZE
    print(f"Total number of samples (clean + realistic): ~{total_images}")

    # Split the combined dataset into training and validation sets
    DATASET_SIZE = tf.data.experimental.cardinality(combined_ds).numpy() * BATCH_SIZE
    TRAIN_SIZE = int(0.8 * DATASET_SIZE)
    VAL_SIZE = DATASET_SIZE - TRAIN_SIZE

    # Shuffle the combined dataset before splitting
    combined_ds = combined_ds.shuffle(buffer_size=DATASET_SIZE)

    # Convert to NumPy for easier slicing 
    def to_np(dataset):
        images, labels = [], []
        for img_batch, label_batch in dataset:
            images.append(img_batch.numpy())
            labels.append(label_batch.numpy())
        return np.concatenate(images), np.concatenate(labels)

    images, labels = to_np(combined_ds.unbatch().batch(DATASET_SIZE))
    
    x_train = images[:TRAIN_SIZE]
    y_train = labels[:TRAIN_SIZE]
    x_val = images[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    y_val = labels[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]

    # Normalize pixel values to the range [0, 1]
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    
    return (x_train, y_train), (x_val, y_val), class_names, num_classes


# --- MODEL DEFINITION ---
def build_cnn_model(num_classes):
    """Defines a simple CNN architecture for orbit classification."""
    model = models.Sequential([
        
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), 
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# --- TRAINING EXECUTION ---
if __name__ == '__main__':
    try:
        # Load data
        (x_train, y_train), (x_val, y_val), class_names, num_classes = load_and_combine_data()

        # Build and train model
        model = build_cnn_model(num_classes)
        model.summary()
        
        print("\nStarting model training...")
        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        # Save the model
        model.save(MODEL_FILE)
        print(f"\nTraining complete. Model saved to {MODEL_FILE}")
        
        with open("class_names.txt", "w") as f:
            f.write(','.join(class_names))
        print("Class names saved to class_names.txt")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure your data generation script has run successfully and created the 'orbit_dataset' folder.")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")