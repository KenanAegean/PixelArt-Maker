import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from rembg import remove
from PIL import Image, ImageTk
import io
import os
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Global variable to store the image path
image_path = None

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

# Function to display the processed image in the GUI
def display_image(image_array, label):
    img_pil = Image.fromarray(image_array)
    img_pil = img_pil.resize((200, 200), Image.Resampling.LANCZOS)  # Resize to fit in the GUI
    img_tk = ImageTk.PhotoImage(img_pil)
    label.config(image=img_tk)
    label.image = img_tk  # Keep reference to avoid garbage collection

# Function to save the output image and add a number if file exists
def save_image(output_path, image_data):
    # Check if the file already exists, and if so, append a number to the filename
    base, ext = os.path.splitext(output_path)
    counter = 1
    new_output_path = output_path

    while os.path.exists(new_output_path):
        new_output_path = f"{base}_{counter}{ext}"
        counter += 1

    resized_image = cv2.resize(image_data, (512, 512), interpolation=cv2.INTER_AREA)
    Image.fromarray(resized_image).save(new_output_path, format='PNG')
    print(f"Output saved to {new_output_path}")
    return new_output_path  # Return the path of the saved image

# Function to process the image and generate pixel art
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

        # Step 6: Save the output
        saved_image_path = save_image(output_path, pixel_art)

        # Step 7: Load and display the saved image (512x512)
        saved_image = Image.open(saved_image_path)
        saved_image_resized = saved_image.resize((200, 200), Image.Resampling.LANCZOS)  # Resize for display
        img_tk = ImageTk.PhotoImage(saved_image_resized)
        generated_image_label.config(image=img_tk)
        generated_image_label.image = img_tk  # Keep reference to avoid garbage collection
    else:
        messagebox.showerror("Error", "No face detected in the image!")

# Tkinter GUI Setup
# Function to handle image selection
def open_file():
    global image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        image_path = file_path  # Save the selected image path
        img = Image.open(file_path)
        img = img.resize((200, 200), Image.Resampling.LANCZOS)  # Use LANCZOS for resizing in newer Pillow versions
        img_tk = ImageTk.PhotoImage(img)
        selected_image_label.config(image=img_tk)
        selected_image_label.image = img_tk  # Save reference to avoid garbage collection
        
        # Get the file name and update the button text
        file_name = os.path.basename(file_path)
        select_image_button.config(text=file_name)  # Update the button text with the selected image's filename
    else:
        messagebox.showerror("Error", "No image selected!")

# Function to generate pixel art
def generate_art():
    if image_path is None:
        messagebox.showerror("Error", "Please select an image first!")
        return

    try:
        num_colors = int(color_entry.get())
        pixel_size = int(pixel_size_entry.get())
        output_path = "output/pixel_art.png"

        # Run the pixel art processing
        process_image(image_path, output_path, num_colors, pixel_size)
        messagebox.showinfo("Success", "Pixel art generated successfully!")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for colors and pixel size!")

# Initialize the root window
root = tk.Tk()
root.title("PixelArt Generator")
root.geometry("900x600")
root.config(bg="#444")

# Title label
title_label = tk.Label(root, text="PixelArt Generator", font=("Arial", 32, "bold"), bg="#444", fg="white")
title_label.pack(pady=20)

# Main content frame
frame = tk.Frame(root, bg="#444")
frame.pack(pady=20)

# Left Frame for image selection and input fields
left_frame = tk.Frame(frame, bg="#444")
left_frame.pack(side=tk.LEFT, padx=20)

# Select Image Button
select_image_button = tk.Button(left_frame, text="Drag or select", font=("Arial", 12, "bold"), bg="#666", fg="white", width=20, height=1, relief="flat")
select_image_button.pack(pady=10)
select_image_button.config(command=open_file)

# File label
file_label = tk.Label(left_frame, text="Select Image", font=("Arial", 14), bg="#444", fg="white")
file_label.pack(pady=5)

# Number of Colors Input
color_label = tk.Label(left_frame, text="Number Of Color:", font=("Arial", 12), bg="#444", fg="white")
color_label.pack(pady=5)
color_entry = tk.Entry(left_frame, font=("Arial", 12), width=20, relief="flat", bg="#666", fg="white")
color_entry.pack(pady=5)
color_entry.insert(0, "8")  # Default value

# Pixel Size Input
pixel_size_label = tk.Label(left_frame, text="Pixel Size (e.g., 32, 48)", font=("Arial", 12), bg="#444", fg="white")
pixel_size_label.pack(pady=5)
pixel_size_entry = tk.Entry(left_frame, font=("Arial", 12), width=20, relief="flat", bg="#666", fg="white")
pixel_size_entry.pack(pady=5)
pixel_size_entry.insert(0, "48")  # Default value

# Generate Button
generate_button = tk.Button(left_frame, text="GENERATE", font=("Arial", 14, "bold"), bg="#666", fg="white", width=20, height=2, relief="flat")
generate_button.pack(pady=20)
generate_button.config(command=generate_art)

# Right Frame for displaying the images
right_frame = tk.Frame(frame, bg="#444")
right_frame.pack(side=tk.LEFT, padx=40)

# Label for selected image
selected_image_label_title = tk.Label(right_frame, text="Selected Image:", font=("Arial", 14), bg="#444", fg="white")
selected_image_label_title.pack()

# Placeholder for selected image
selected_image_label = tk.Label(right_frame, bg="#666", width=200, height=200)
selected_image_label.pack(pady=10)

# Label for generated image
generated_image_label_title = tk.Label(right_frame, text="Generated Image:", font=("Arial", 14), bg="#444", fg="white")
generated_image_label_title.pack()

# Placeholder for generated image
generated_image_label = tk.Label(right_frame, bg="#666", width=200, height=200)
generated_image_label.pack(pady=10)

# Footer credits
footer_label = tk.Label(root, text="by Kaegean", font=("Arial", 14, "bold"), bg="#444", fg="white")
footer_label.pack(side=tk.BOTTOM, pady=10)

# Start the Tkinter main loop
root.mainloop()

