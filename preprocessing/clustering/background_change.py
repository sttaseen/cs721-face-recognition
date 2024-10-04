import os
import cv2
import numpy as np
from PIL import Image
import random
from glob import glob
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load Mask-RCNN for background removal
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Helper function to apply the mask and replace the background
def apply_background_change(image, mask, background):
    # Resize the background to the same size as the input image
    background = background.resize(image.size)
    
    # Convert image and background to numpy arrays
    image_np = np.array(image)
    background_np = np.array(background)
    
    # Create a 3-channel mask
    mask = mask[:, :, np.newaxis]  # Add extra dimension to match image channels
    mask = np.repeat(mask, 3, axis=2)  # Convert single-channel mask to 3-channel

    # Apply the mask
    result = np.where(mask, image_np, background_np)

    return Image.fromarray(result)

# Function to replace background in an image
def replace_background(image_path, background_image=None):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Convert to PyTorch tensor and normalize
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image_np).unsqueeze(0)

    # Run the Mask R-CNN model
    with torch.no_grad():
        prediction = model(image_tensor)

    # Get the mask for the largest detected object
    masks = prediction[0]['masks']
    if len(masks) > 0:
        mask = masks[0, 0].mul(255).byte().cpu().numpy()
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        
        if background_image is None:
            # Generate a random background (for data augmentation)
            background = Image.fromarray(np.uint8(np.random.rand(*image_np.shape) * 255))
        else:
            # Load the provided background image
            background = Image.open(background_image).convert('RGB')

        # Apply the mask and replace the background
        result_image = apply_background_change(image, mask, background)
        
        return result_image
    else:
        print(f"No object detected in {image_path}")
        return image

# Process all images in a folder
def process_folder(input_folder, output_folder, background_folder=None):
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in the input folder (supporting multiple image extensions)
    image_files = glob(f"{input_folder}/*.*")  # Grab all files in the input folder
    image_files = [file for file in image_files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    background_images = glob(f"{background_folder}/*.jpg") if background_folder else None

    print(f"Background images: {background_images}")
    print(f"Image files: {image_files}")

    for image_file in image_files:
        print(f"Processing {image_file}...")
        # Choose a random background image if a folder is provided
        if background_images:
            background_image = random.choice(background_images)
        else:
            background_image = None
        
        # Replace the background
        new_image = replace_background(image_file, background_image)
        
        # Save the result
        output_path = os.path.join(output_folder, os.path.basename(image_file))
        new_image.save(output_path)

# Example usage:
input_folder = "./clusters"
output_folder = "./background_changed"
background_folder = "./backgrounds"  

process_folder(input_folder, output_folder, background_folder)
