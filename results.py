import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os

# Paths
validation_dir = r'C:\Users\ABHISHEK\Downloads\projects\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid'
model_path = 'crop_disease_model.keras'
class_indices_path = 'class_indices.json'

# Load the trained model and class indices
if os.path.exists(model_path) and os.path.exists(class_indices_path):
    print("Loading existing model and class indices...")
    
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load class indices
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Reverse the class indices for label mapping
    label_map = {v: k for k, v in class_indices.items()}

    # Create the ImageDataGenerator for validation data
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation Loss: {loss:.4f}')
    print(f'Validation Accuracy: {accuracy:.4f}')

    # Predict on the validation set
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    # Generate confusion matrix and classification report
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))

    print('Classification Report')
    target_names = list(label_map.values())
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

else:
    print("Model or class indices file not found!")
