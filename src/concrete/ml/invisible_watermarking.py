import numpy as np
from concrete import *
import cv2
from sklearn.preprocessing import MinMaxScaler

# Function to load an image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return image

# Function to encode watermark into the image using LSB (Least Significant Bit) method
def encode_watermark(image, watermark):
    watermark = cv2.resize(watermark, (image.shape[1], image.shape[0]))  # Resize watermark to fit image size
    encoded_image = np.copy(image)

    # Iterate over the image and embed the watermark
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Set the LSB to the corresponding watermark bit
            encoded_image[i, j] = (encoded_image[i, j] & 0xFE) | (watermark[i, j] >> 7)
    return encoded_image

# Function to extract watermark (using LSB)
def extract_watermark(encoded_image):
    watermark = np.copy(encoded_image)
    for i in range(encoded_image.shape[0]):
        for j in range(encoded_image.shape[1]):
            watermark[i, j] = (encoded_image[i, j] & 0x01) << 7
    return watermark

# Function to perform FHE-based encryption of the image
def encrypt_image(image):
    # Set up the FHE encryption scheme using Concrete ML
    context = Context()  # Setup context for Concrete FHE
    encryptor = Encryptor(context)

    # Scale image to 8-bit format before encryption
    scaler = MinMaxScaler(feature_range=(0, 255))
    image_scaled = scaler.fit_transform(image.astype(np.float32).reshape(-1, 1)).reshape(image.shape)

    encrypted_image = encryptor.encrypt(image_scaled)
    return encrypted_image

# Function to decrypt an encrypted image
def decrypt_image(encrypted_image):
    decryptor = Decryptor()
    decrypted_image = decryptor.decrypt(encrypted_image)

    # Rescale back to the original pixel range
    scaler = MinMaxScaler(feature_range=(0, 255))
    image_decrypted = scaler.inverse_transform(decrypted_image.reshape(-1, 1)).reshape(decrypted_image.shape)
    return image_decrypted.astype(np.uint8)

# Example of using FHE to encode and decode images with watermark
def main():
    # Load the image and watermark
    image = load_image('path_to_image.jpg')
    watermark = load_image('path_to_watermark.png')

    # Encode the watermark in the image
    encoded_image = encode_watermark(image, watermark)

    # Encrypt the image with watermark
    encrypted_image = encrypt_image(encoded_image)

    # Decrypt the image
    decrypted_image = decrypt_image(encrypted_image)

    # Extract the watermark from the decrypted image
    extracted_watermark = extract_watermark(decrypted_image)

    # Show the images
    cv2.imshow('Original Image', image)
    cv2.imshow('Watermarked Image', encoded_image)
    cv2.imshow('Extracted Watermark', extracted_watermark)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
