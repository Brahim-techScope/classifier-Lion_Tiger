import rembg
from PIL import Image
import os

# Define input and output directories
input_dir = 'lion_tiger/train/tigers/'
output_dir = 'lion_tiger/train/tigers/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through the files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, 'a' + filename.split('.')[0] + '.png')  # Change the extension to .png
        
        # Open the input image
        inp = Image.open(input_path)
        
        # Remove the background
        output = rembg.remove(inp)
        
        # Save the background-removed image
        output.save(output_path)
print("Background removal complete.")
