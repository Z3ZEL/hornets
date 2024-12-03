import kagglehub
import torch
import cv2
from ultralytics import YOLO
from torchvision import datasets, transforms
import kagglehub




def main():
# Download latest version
    path = kagglehub.dataset_download("jerzydziewierz/bee-vs-wasp")

    print("Path to dataset files:", path)
    
    # Load the model
    model = YOLO("yolo-Weights/yolo11n.pt")
    
    # Load the dataset
    dataset = datasets.ImageFolder(path, transform=transforms.Compose([transforms.ToTensor()]))
    
    # Load the data
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Train the model
    model.train(data=data_loader, epochs=1, batch=64)

    # Save the model    
    model.save("trained_model.pt")  
