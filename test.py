import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# Define paths to your dataset
train_dir = '/Users/krishteenun/Documents/Edited_dataset/train'
test_dir = '/Users/krishteenun/Documents/Edited_dataset/test'
valid_dir = '/Users/krishteenun/Documents/Edited_dataset/valid'

# Load the model
model = tf.keras.models.load_model('model_05.keras')

# Data generators for loading and preprocessing images
datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  # Adjust this to your model's expected input size
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Make predictions
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int)

# Get the true labels from the test generator
true_classes = test_generator.classes

# Calculate metrics
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='binary')
recall = recall_score(true_classes, predicted_classes, average='binary')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
