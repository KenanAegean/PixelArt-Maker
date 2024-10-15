import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from rembg import remove
from PIL import Image
import io
import os
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Function to remove the background using rembg
def remove_background(image_path):
    with open(image_path, 'rb') as input_file:
        input_data = input_file.read()
    output_data = remove(input_data)
    img_np = np.array(Image.open(io.BytesIO(output_data)))
    return img_np

# Function to detect and crop the face
def detect_and_crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
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

# Function to resize the cropped face
def resize_to_square(image, size):
    resized_face = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return resized_face

# Function to reduce the color palette using KMeans clustering
def quantize_image(image, num_colors):
    pixels = image.reshape(-1, 4)
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(pixels)
    new_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    quantized_image = new_colors[labels].reshape(image.shape).astype(np.uint8)
    return quantized_image

# Function to find the most frequent color in a given square
def most_frequent_color(square):
    pixels = square.reshape(-1, square.shape[2])
    pixels = [tuple(pixel) for pixel in pixels]
    color_counter = Counter(pixels)
    most_common_color = color_counter.most_common(1)[0][0]
    return most_common_color

# Function to create pixel art
def create_pixel_art(image, square_size=1):
    height, width, _ = image.shape
    pixel_art = np.zeros_like(image)

    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            square = image[y:y+square_size, x:x+square_size]
            dominant_color = most_frequent_color(square)
            pixel_art[y:y+square_size, x:x+square_size] = dominant_color

    return pixel_art

# Function to save the output image
def save_image(output_path, image_data):
    resized_image = cv2.resize(image_data, (512, 512), interpolation=cv2.INTER_AREA)
    Image.fromarray(resized_image).save(output_path, format='PNG')
    print(f"Output saved to {output_path}")

# Function to process the image
def process_image(image_path, output_path, num_colors, pixel_size):
    # Step 1: Remove background
    image_no_bg = remove_background(image_path)

    # Step 2: Detect and crop face
    cropped_face = detect_and_crop_face(image_no_bg)
    if cropped_face is not None:
        # Step 3: Resize face
        resized_face = resize_to_square(cropped_face, size=pixel_size)

        # Step 4: Quantize colors
        quantized_face = quantize_image(resized_face, num_colors)

        # Step 5: Create pixel art
        pixel_art = create_pixel_art(quantized_face, square_size=1)

        # Step 6: Save image
        save_image(output_path, pixel_art)
    else:
        messagebox.showerror("Error", "No face detected in the image!")

# Tkinter GUI Setup
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        file_label.config(text=file_path)
        return file_path
    return None

def generate_art():
    image_path = file_label.cget("text")
    if not image_path:
        messagebox.showerror("Error", "Please select an image file!")
        return

    # Get user inputs for color number and pixel size
    num_colors = int(color_entry.get())
    pixel_size = int(pixel_size_entry.get())
    output_path = "output/pixel_art.png"

    # Run the pixel art processing
    process_image(image_path, output_path, num_colors, pixel_size)

# Tkinter Window
root = tk.Tk()
root.title("Pixel Art Generator")

# File selection
file_label = tk.Label(root, text="No file selected")
file_label.pack()
file_button = tk.Button(root, text="Select Image", command=open_file)
file_button.pack()

# Color number selection
color_label = tk.Label(root, text="Number of Colors:")
color_label.pack()
color_entry = tk.Entry(root)
color_entry.pack()
color_entry.insert(0, "8")  # Default value

# Pixel size selection
pixel_size_label = tk.Label(root, text="Pixel Size (e.g., 48, 32):")
pixel_size_label.pack()
pixel_size_entry = tk.Entry(root)
pixel_size_entry.pack()
pixel_size_entry.insert(0, "48")  # Default value

# Generate button
generate_button = tk.Button(root, text="Generate Pixel Art", command=generate_art)
generate_button.pack()

# Run the Tkinter GUI
root.mainloop()
