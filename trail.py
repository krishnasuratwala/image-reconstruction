import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to preprocess a single image to the required size
def preprocess_image(img_path, target_size=(256, 256)):
    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img

# Define the autoencoder architecture
input_img = Input(shape=(256, 256, 3))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Combine encoder and decoder into an autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Data generator for training
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)  # Split data into training and validation sets

train_generator = data_gen.flow_from_directory(
    'caltech-256',  # Path to your dataset directory
    target_size=(256, 256),
    batch_size=32,
    class_mode='input',
    subset='training'
)

validation_generator = data_gen.flow_from_directory(
    'caltech-256',  # Path to your dataset directory
    target_size=(256, 256),
    batch_size=32,
    class_mode='input',
    subset='validation'
)

# Train the autoencoder
autoencoder.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Save the trained model
autoencoder.save('autoencoder_model.h5')

# Load and preprocess the input image
input_image_path = 'input_image.avif'  # Replace 'input_image.jpg' with your input image file
input_image = preprocess_image(input_image_path)

# Add batch dimension
input_image = np.expand_dims(input_image, axis=0)

# Feed the input image to the autoencoder and get the reconstructed image
reconstructed_image = autoencoder.predict(input_image)

# Postprocess the reconstructed image
reconstructed_image = (reconstructed_image * 255).astype(np.uint8)

# Display the input and reconstructed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_image[0])
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image[0])
plt.title('Reconstructed Image')
plt.axis('off')

plt.show()
