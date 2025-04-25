# Install necessary packages (only needed once in your environment)
# pip install opencv-python tensorflow matplotlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Load validation dataset
validation_set = tf.keras.utils.image_dataset_from_directory(
    'archive/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True
)

class_names = validation_set.class_names
print("Class names:", class_names)

# Load the trained CNN model
cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Test image visualization
image_path = '/home/qambar/Desktop/LeafLens-main/archive/test/test/AppleCedarRust1.JPG'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display image
plt.imshow(img)
plt.title("Test Image")
plt.axis('off')
plt.show()

# Resize and expand dimensions to match model input
img_resized = cv2.resize(img, (128, 128))
img_array = np.expand_dims(img_resized, axis=0)

# Prediction
prediction = cnn.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
print("Predicted class:", predicted_class)
