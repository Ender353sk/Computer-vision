import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2
import numpy as np
import os

# Параметри моделі
image_size = (128, 128)
batch_size = 32
classes = ['copper', 'aluminum', 'steel']

# Підготовка даних
data_dir = "path_to_your_dataset"  # Замініть на шлях до вашого набору даних
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    classes=classes,
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    classes=classes,
    subset='validation'
)

# Створення моделі
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Тренування моделі
model.fit(train_data, validation_data=val_data, epochs=10)

# Збереження моделі
model.save("metal_classifier_model.h5")

# Використання моделі
def classify_image(image_path):
    model = tf.keras.models.load_model("metal_classifier_model.h5")
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size) / 255.0
    image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    class_index = np.argmax(predictions)
    print(f"Classified as: {classes[class_index]}")

# Приклад використання
sample_image = "path_to_sample_image.jpg"  # Замініть на шлях до вашого зображення
classify_image(sample_image)
