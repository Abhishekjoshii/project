import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory paths (update if necessary)
train_dir = r'C:\Users\ABHISHEK\Downloads\projects\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train'

# Create Image Data Generator for loading class indices
train_datagen = ImageDataGenerator()

# Loading data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Save class indices to JSON file
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as json_file:
    json.dump(class_indices, json_file)
