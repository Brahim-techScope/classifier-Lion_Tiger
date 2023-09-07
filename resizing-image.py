from PIL import Image
import os

# Define the source and destination directories
source_dir = 'lion_tiger/train/lions'
destination_dir = 'lion_tiger/train/lions'

# Define the target size (300x300) and number of color channels (3 for RGB)
target_size = (300, 300)
num_channels = 3

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Iterate through the images in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # Open the image
        img = Image.open(os.path.join(source_dir, filename))

        # Resize the image to the target size and convert it to RGB
        img = img.resize(target_size, Image.LANCZOS).convert('RGB')

        # Save the resized image to the destination directory
        img.save(os.path.join(destination_dir, filename))
print("Resizing and saving complete.")
