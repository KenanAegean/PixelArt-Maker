
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans

# Function to detect and crop the face
def detect_and_crop_face(image_path):
    # Load the pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No face detected.")
        return None

    # Assuming we only want the first detected face
    (x, y, w, h) = faces[0]

    # Add some padding around the face for cropping
    padding = int(0.2 * w)  # Add 20% padding
    x, y = max(0, x-padding), max(0, y-padding)
    w, h = w + 2*padding, h + 2*padding

    # Crop the face with padding
    cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Function to resize the cropped face to a smaller grid
def resize_to_square(image, size=48):
    resized_face = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return resized_face

# Function to reduce the color palette using KMeans clustering
def quantize_image(image, num_colors=10):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(pixels)
    new_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    quantized_image = new_colors[labels].reshape(image.shape).astype(np.uint8)
    return quantized_image

# Function to find the most frequent color in a given square
def most_frequent_color(square):
    pixels = square.reshape(-1, 3)
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

# Function to enhance edges and smooth color transitions
def enhance_edges_sharper(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)  # Less blur for sharper edges
    edges = cv2.Canny(blurred, threshold1=50, threshold2=120)  # Sharper edges
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    enhanced_image = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)
    smoothed_image = cv2.GaussianBlur(enhanced_image, (1, 1), 0)  # Slight smoothing
    return smoothed_image

# Final function to process the image with 48x48 size and a maximum of 10 colors
def process_image_with_10_colors(image_path, square_size=48):
    cropped_face = detect_and_crop_face(image_path)
    if cropped_face is not None:
        # Enhance edges with sharper detection
        enhanced_face = enhance_edges_sharper(cropped_face)
        resized_face = resize_to_square(enhanced_face, size=48)  # Resize to exactly 48x48
        # Apply default KMeans color quantization with 10 colors
        quantized_face = quantize_image(resized_face, num_colors=10)
        # Create pixel art with square size of 1 (since we resized to 48x48)
        pixel_art = create_pixel_art(quantized_face, square_size=1)  # Each block should represent 1 pixel of 48x48
        # Convert to RGB for display
        pixel_art_rgb = cv2.cvtColor(pixel_art, cv2.COLOR_BGR2RGB)
        # Display the result
        plt.imshow(pixel_art_rgb)
        plt.axis('off')  # Hide axes
        plt.show()
    else:
        print("Face not detected.")

# Example usage:
# process_image_with_10_colors('your_image_path_here.jpg')
