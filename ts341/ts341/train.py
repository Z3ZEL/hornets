import kagglehub
import torch
import cv2
from ultralytics import YOLO
from torchvision import datasets, transforms
# import kagglehub




def main():
    # Download latest version
    # path = kagglehub.dataset_download("marcoryvandijk/vespa-velutina-v-crabro-vespulina-vulgaris")

    # path = "/home/louis/Documents/3A/Outils d'imagerie pour la robotique/TD/hornets/ts341/datasets/dataset_2"

    # print("Path to dataset files:", path)
    
    # Load the model
    model = YOLO("yolo-Weights/yolo11n.pt")
    
    # transform = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor()  # Convertit en tenseur
    # ])

    # # Load the dataset
    # dataset = datasets.ImageFolder(root=path, transform=transform)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # print("Mon data_loader")

    # Train the model
    model.train(data="/home/louis/Documents/3A/Outils_imagerie/TD/hornets/ts341/datasets/dataset_3/vespa-velutina-v-crabro-vespulina-vulgaris/versions/8/config.yaml", epochs=1, batch=32, imgsz=640, save=True)
