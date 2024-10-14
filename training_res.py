import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import time
import csv

# Define transformations for the ImageNet dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet dataset (using the validation set for inference)
imagenet_val = ImageNet(root='path/to/imagenet', split='val', transform=transform)
val_loader = DataLoader(imagenet_val, batch_size=1, shuffle=True)

# Define the models to be evaluated
model_names = ['resnet', 'vgg', 'inception', 'efficientnet', 'alexnet', 'mobilenet']
models_dict = {
    'resnet': models.resnet50(pretrained=True),
    'vgg': models.vgg16(pretrained=True),
    'inception': models.inception_v3(pretrained=True),
    'efficientnet': models.efficientnet_b0(pretrained=True),
    'alexnet': models.alexnet(pretrained=True),
    'mobilenet': models.mobilenet_v2(pretrained=True)
}

# Set the models to evaluation mode and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model in models_dict.values():
    model.to(device)
    model.eval()

# Function to compute accuracy
def compute_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    return (predicted == target).sum().item()

# Open CSV file for writing the results
with open('res.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['model', 'image_id', 'accuracy', 'inference_time'])

    # Loop through the data loader for inference
    image_count = 0
    for images, labels in val_loader:
        if image_count >= 100000:  # Stop after 100k images
            break

        images, labels = images.to(device), labels.to(device)

        # Run inference on each model and record results
        for model_name, model in models_dict.items():
            start_time = time.time()
            with torch.no_grad():
                outputs = model(images)
            end_time = time.time()

            # Calculate accuracy
            accuracy = compute_accuracy(outputs, labels)

            # Calculate inference time
            inference_time = end_time - start_time

            # Write results to CSV
            writer.writerow([model_name, image_count, accuracy, inference_time])

        image_count += 1

print("Inference complete. Results saved to res.csv")
