import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from rembg import remove
from PIL import Image
import io
import os

# Define input and output paths
image_path = 'images/input_image2.jpg'   # Input image path
output_path = 'output/pixel_art.png'    # Output path to save the pixel art

# Function to remove the background using rembg
def remove_background(image_path):
    with open(image_path, 'rb') as input_file:
        input_data = input_file.read()
    output_data = remove(input_data)
    # Convert output to a transparent image (RGBA) using OpenCV
    img_np = np.array(Image.open(io.BytesIO(output_data)))
    return img_np  # Return image with transparency (RGBA format)

# Function to detect and crop the face
def detect_and_crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)  # Convert to grayscale (for transparent images)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No face detected.")
        return None

    (x, y, w, h) = faces[0]
    padding = int(0.2 * w)  
    x, y = max(0, x-padding), max(0, y-padding)
    w, h = w + 2*padding, h + 2*padding

    cropped_face = image[y:y+h, x:x+w]
    return cropped_face

# Function to resize the cropped face to a smaller grid
def resize_to_square(image, size=48):
    resized_face = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return resized_face

# Function to find the most frequent color in a given square
def most_frequent_color(square):
    pixels = square.reshape(-1, square.shape[2])
    pixels = [tuple(pixel) for pixel in pixels]
    color_counter = Counter(pixels)
    most_common_color = color_counter.most_common(1)[0][0]
    return most_common_color

# Function to create pixel art by applying dominant color to mini-squares
def create_pixel_art(image, square_size=1):
    height, width, _ = image.shape
    pixel_art = np.zeros_like(image)

    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            square = image[y:y+square_size, x:x+square_size]
            dominant_color = most_frequent_color(square)
            pixel_art[y:y+square_size, x:x+square_size] = dominant_color

    return pixel_art

# Function to save the output image with a number if the file already exists
def save_with_incrementing_filename(output_path, image_data):
    base_name, ext = os.path.splitext(output_path)
    counter = 1
    new_output_path = output_path

    # Check if the file exists and create a new name with a number if necessary
    while os.path.exists(new_output_path):
        new_output_path = f"{base_name}_{counter}{ext}"
        counter += 1

    # Save the image as 512x512
    resized_image = cv2.resize(image_data, (512, 512), interpolation=cv2.INTER_AREA)
    Image.fromarray(resized_image).save(new_output_path, format='PNG')
    print(f"Output saved to {new_output_path}")

# Final function to process the image with background removal and pixel art
def process_image_with_bg_removal(image_path, output_path, square_size=48):
    # Step 1: Remove background
    image_no_bg = remove_background(image_path)
    
    # Step 2: Detect and crop face
    cropped_face = detect_and_crop_face(image_no_bg)
    if cropped_face is not None:
        # Step 3: Resize face
        resized_face = resize_to_square(cropped_face, size=48)  # Resize to 48x48
        pixel_art = create_pixel_art(resized_face, square_size=1)  # Create pixel art with block size of 1

        # Convert RGBA to RGB for display and saving
        pixel_art_rgba = cv2.cvtColor(pixel_art, cv2.COLOR_RGB2RGBA)
        
        # Display the result
        plt.imshow(pixel_art_rgba)
        plt.axis('off')  # Hide axes
        plt.show()
        
        # Save the output image with file name incrementing logic and resized to 512x512
        save_with_incrementing_filename(output_path, pixel_art_rgba)
    else:
        print("Face not detected.")

# Process the image and save the result
process_image_with_bg_removal(image_path, output_path)
