
# Pixel Art Generator

This project is a Python-based pixel art generator that processes an input image (preferably of a face), removes the background, and creates a pixelated version of the image. The final output is resized to 512x512 pixels with a transparent background.

## Features
- **Background Removal**: Automatically removes the background using the rembg library.
- **Face Detection**: Automatically detects the face in the image using OpenCV's Haar Cascade classifier.
- **Pixelation**: Divides the image into squares and fills each square with the most frequent color, creating a pixelated effect.
- **Transparent Background**: The final pixel art retains a transparent background.
- **Auto File Renaming**: If the output file already exists, a number is appended to avoid overwriting existing files.
- **Final Output Resized**: The final output image is resized to 512x512 pixels.

## Project Structure

```
your-repo-name/
│
├── pixel_art.py           # Main script for generating pixel art
├── requirements.txt       # Python dependencies
├── README.md              # Documentation for the project
├── images/                # Folder to store input and output images (optional)
│   └── input_image.png    # Example input image (optional)
├── output/                # Folder to store input and output images (optional)
│   └── pixel_art.png   # Example pixel art output (optional)
└── .gitignore             # Ignore unnecessary files in the repo (optional)
```

## Installation

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/KenanAegean/PixelArt-Maker.git
cd PixelArt-Maker
```

### 2. Install Dependencies
You can install the required Python dependencies using `pip`:

```bash
pip install -r requirements.txt
```

This will install the following packages:
- `opencv-python`: For image processing and face detection.
- `matplotlib`: For displaying images.
- `rembg`: For background removal.
- `Pillow`: For saving images.

## Usage

To generate pixel art from an image, follow these steps:

### 1. Prepare Your Image
Ensure you have an image you want to convert into pixel art. Place this image in the `images/` folder (or any location of your choice).

### 2. Modify the Script
In the `pixel_art.py` script, update the `image_path` variable with the path to your input image:
```python
image_path = 'images/input_image.jpg'
```

### 3. Run the Script
Run the `pixel_art.py` script to generate the pixel art:

```bash
python pixel_art.py
```

This will display the generated pixel art on your screen.

### 4. Saving the Output
The final output is automatically saved as a PNG with a transparent background, and it is resized to 512x512 pixels. If a file with the same name exists, the script appends a number to avoid overwriting.

### Example Usage

```python
# Example of using the process_image function
image_path = 'images/input_image.jpg'
output_path = 'output/pixel_art.png'
process_image_with_bg_removal(image_path, output_path)
```

### Output
The pixel art image will be displayed on your screen, and saved in the `output/` folder. If the file already exists, a new file with a number (e.g., `pixel_art_1.png`) will be created.

## Example
**GUI:**

![GUI Image](for_readme/picture_of_gui.png)

Before running the script, you can place an image like this in the `images/` folder:

**Input:**

![Input Image](for_readme/input.jpg)

**Output:**

After running the script, you will get pixel art output similar to this:

![Output Image](for_readme/output.png)

## How It Works

1. **Background Removal**: The script uses the `rembg` library to remove the background from the input image.
2. **Face Detection**: The script detects a face in the input image using OpenCV's Haar Cascade classifier.
3. **Pixelation**: The image is divided into square blocks, and the most frequent color in each block is applied, creating the pixelated look.
4. **Final Resizing**: The output image is resized to 512x512 pixels and saved with a transparent background.

## Customization

### Adjusting Pixelation
You can modify the square size used in the pixelation by adjusting the `square_size` parameter in the script.

### Saving the Output with Transparency
The output is automatically saved as a PNG with a transparent background. If needed, you can modify the `output_path` variable in the script to specify a different output location.

