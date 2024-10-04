import os
import cv2
import torch
import numpy as np
from sklearn.cluster import KMeans
from torchvision import models, transforms
import torch.nn as nn
import shutil  # Import to handle file copying

# Step 1: Load images from the folder (without resizing)
def load_images_from_folder(folder_path):
    images = []
    image_files = []
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            image_files.append(filename)
    
    return np.array(images), image_files

# Step 2: Extract features using a pre-trained VGG16 model with PyTorch
def extract_features_pytorch(images):
    # Load the pre-trained VGG16 model
    model = models.vgg16(pretrained=True)
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])  # Remove the last layers
    model.eval()  # Set model to evaluation mode
    
    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Extract features for all images
    features = []
    for img in images:
        img = preprocess(img)
        img = img.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            feature_vector = model(img).numpy().flatten()  # Get the output feature vector
        
        features.append(feature_vector)
    
    return np.array(features)

# Step 3: Perform KMeans clustering to group images
def cluster_images(features, n_clusters=20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    
    return kmeans.labels_

# Step 4: Copy representative images from each cluster
def save_cluster_representatives(image_files, labels, input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        # Get the indices of images in this cluster
        indices = np.where(labels == label)[0]
        representative_idx = indices[0]  # Choose the first image as the representative
        
        # Get the original image file path and copy to output folder
        img_filename = image_files[representative_idx]
        src_path = os.path.join(input_folder, img_filename)
        dest_path = os.path.join(output_folder, f"cluster_{label}_{img_filename}")
        
        shutil.copy(src_path, dest_path)  # Copy the image file without resizing

# Main function to execute the steps
def cluster_and_save_images(folder_path, output_folder, n_clusters=20):
    # Load images from folder
    images, image_files = load_images_from_folder(folder_path)
    
    # Extract features from images using PyTorch VGG16
    features = extract_features_pytorch(images)
    
    # Perform clustering
    labels = cluster_images(features, n_clusters=n_clusters)
    
    # Save representative images from each cluster by copying them
    save_cluster_representatives(image_files, labels, folder_path, output_folder)

# Define folder paths
input_folder = './input'  # Path to the folder containing the images
output_folder = './clusters'  # Path to save the representative images

# Execute the process
cluster_and_save_images(input_folder, output_folder, n_clusters=20)
