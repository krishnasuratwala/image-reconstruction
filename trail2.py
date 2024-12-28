import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageOps

# Load the trained autoencoder model
autoencoder = load_model('autoencoder_model.h5')  # Replace 'autoencoder_model.h5' with your model file

# Define the encoder part to extract features
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('max_pooling2d_1').output)

# Function to preprocess a single image to the required size
def preprocess_image(img_path, target_size=(256, 256)):
    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img

# Load and preprocess the input image
input_image_path = 'input_image.jpg'  # Replace 'input_image.jpg' with your input image file
input_image = Image.open(input_image_path)

# Convert the image to RGB mode if it's not already in RGB
if input_image.mode != 'RGB':
    input_image = input_image.convert('RGB')

# Crop the image to a square region around the center
# Calculate padding dimensions
width, height = input_image.size
padding_size = abs(width - height) // 2

# Pad the image to make it square
if width > height:
    input_image = ImageOps.expand(input_image, border=(0, padding_size), fill=0)
else:
    input_image = ImageOps.expand(input_image, border=(padding_size, 0), fill=0)

# Resize the cropped image
input_image = input_image.resize((256, 256))  # Resize to match the input size of the autoencoder

# Convert image to numpy array
input_image = np.array(input_image) / 255.0

# Ensure the input image has the correct shape
if input_image.ndim == 2:  # If the image is grayscale, repeat it along the third axis to create three channels
    input_image = np.repeat(input_image[:, :, np.newaxis], 3, axis=2)

# Add batch dimension
input_image = np.expand_dims(input_image, axis=0)
reconstructed_image = autoencoder.predict(input_image)

# Extract features from the input image using the encoder
encoded_features = encoder.predict(input_image)

# Print the extracted features
print("Extracted Features:")
print(encoded_features)

# Postprocess the reconstructed image
reconstructed_image = (reconstructed_image * 255).astype(np.uint8)  # Convert pixel values back to uint8

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

# Extract features from the input image using the encoder
encoded_features = encoder.predict(input_image)

# Print the extracted features
print("Extracted Features:")
print(encoded_features)

# Save the extracted features to a file
with open('encoded_features.pkl', 'wb') as f:
    pickle.dump(encoded_features, f)

# Compress the file
with open('encoded_features.pkl', 'rb') as f_in:
    with gzip.open('encoded_features.pkl.gz', 'wb') as f_out:
        f_out.writelines(f_in)

print("Encoded features have been saved and compressed.")

# Optionally, remove the uncompressed file
import os
os.remove('encoded_features.pkl')
