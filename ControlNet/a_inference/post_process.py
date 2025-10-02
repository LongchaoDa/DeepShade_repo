import cv2
import matplotlib.pyplot as plt
import numpy as np

# Helper function to convert hex color code to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Color definitions
custom_grey = hex_to_rgb('#9b9b9b')  # Convert hex color to RGB
black = (0, 0, 0)
white = (255, 255, 255)

# Paths to the images
mask_image_path = '/scratch/YOURNAME/project/ControlNet/training/lc/source/3dmodelimage_row0_col0.png'
target_image_path = '/scratch/YOURNAME/project/ControlNet/a_inference/out/epoch507/generated_result_case_6.png'

# Load the images
mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)  # Load mask in grayscale to detect black areas
target_image = cv2.imread(target_image_path)  # Load the target image in color

# Check if the images were loaded successfully
if mask_image is None or target_image is None:
    raise ValueError("One or both images could not be loaded. Check the paths.")

# Resize the target image to match the mask image dimensions if necessary
if target_image.shape[:2] != mask_image.shape[:2]:
    target_image = cv2.resize(target_image, (mask_image.shape[1], mask_image.shape[0]))

# Convert mask image to binary: black areas will be True, everything else False
_, binary_mask = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY_INV)  # Invert to make black as mask

# Create a white image to overlay the black areas in the target image
white_background = np.full_like(target_image, 255)

# Apply mask: Set black areas in the target image to white
combined_image = np.where(binary_mask[..., None] == 255, white_background, target_image)

# Convert combined image to grayscale for final color processing
combined_gray = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)

# Create final processed image with 3 channels
final_image = np.zeros_like(combined_image)

# Assign colors based on grayscale values:
# - Set black (0) as actual black
# - Set white (255) as actual white
# - Set middle tones to custom grey (#9b9b9b)
final_image[combined_gray <= 40] = black           # Black areas
final_image[combined_gray >= 225] = white          # White areas
final_image[(combined_gray > 30) & (combined_gray < 225)] = custom_grey  # Custom grey for middle tones

# Plot the images side by side
plt.figure(figsize=(12, 6))

# Original target image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
plt.title("Target Image")
plt.axis("off")

# Mask image
plt.subplot(1, 3, 2)
plt.imshow(mask_image, cmap='gray')
plt.title("Mask Image")
plt.axis("off")

# Final processed image
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.title("Final Processed Image")
plt.axis("off")

# Save and display the final image
plt.savefig("/scratch/YOURNAME/project/ControlNet/a_inference/post_process.png", dpi=400)
plt.show()
