from ultralytics import YOLO
import cv2
import math 
import sys
import os

## Check path
def get_path():
    if len(sys.argv) < 2:
        print("Not enough arguments --  poetry run extract bag_file output_folder ")
        sys.exit(1)
    bag_file = sys.argv[1]

    output_folder = sys.argv[2] if len(sys.argv) > 2 else "output"


    if not os.path.exists(bag_file):
        print(f"Video file {bag_file} does not exist")
        sys.exit(1)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        

    return bag_file, output_folder


def main():
    video_path, output_folder = get_path()
    cap = cv2.VideoCapture(video_path)
    cap.set(3, 640)
    cap.set(4, 480)

    ### Load YOLO model
    model = YOLO("runs/detect/train25/weights/best.pt")



    

    while True:
        ### Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        ### Image Preprocessing
        
        
        
        
        ### APPLY YOLO
        results = model(frame)
        results.render()


        




        cv2.imshow("Frame", results.imgs[0])
        k = cv2.waitKey(0)
        if k == 27: 
            break
        