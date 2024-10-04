import torch
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import time
import numpy as np
from torchvision import models

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the ResNet18 model
def create_resnet_model(num_classes):
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43636918, 0.38563913, 0.34477144],
                         std=[0.29639485, 0.2698132, 0.26158142])
])

# Load the trained model
model = create_resnet_model(num_classes=10)  # Modify num_classes as per your use case
model.load_state_dict(torch.load('resnet_34_pretrained.pth'))  # Load the saved model
model.eval()

# Function to get frames per second (FPS)
def get_fps(start_time, frame_count):
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    return fps

# Function to run inference on a single frame
def run_inference(model, frame):
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    input_image = transform(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        start_inference = time.time()
        outputs = model(input_image)
        inference_time = time.time() - start_inference
    
    predicted_class = outputs.argmax(1).item()
    return predicted_class, inference_time

"""
NOTE:
Change the capture device index to the one you want to use.
"""
cap = cv2.VideoCapture(0)

# Variables to track FPS
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference
    predicted_class, inference_time = run_inference(model, frame)
    
    # Update FPS
    frame_count += 1
    fps = get_fps(start_time, frame_count)

    # Display the predicted class, FPS, and inference time on the frame
    cv2.putText(frame, f'Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Inference time: {inference_time:.4f}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Webcam Inference', frame)

    # Exit loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
