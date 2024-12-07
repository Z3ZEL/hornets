from ultralytics import YOLO
from hornets.MOG import filtre_2
import cv2
import sys
import os
import cv2 as cv

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

    ### Load YOLO model
    # model = YOLO("runs/detect/train11/weights/best.pt")
    model = YOLO("../weight/best.pt")

    fgbg = cv.createBackgroundSubtractorMOG2()


    

    while True:
        ### Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        ### Apply Filters
        frame = cv2.resize(frame, (640, 640))
        
        frame = filtre_2(frame, fgbg)
        
        
        ### APPLY YOLO and render bounding boxes
        results = model.predict(frame)


        ### Show frame with bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{'bee' if box.cls.item() == 1 else 'hornet' } {box.conf.item():.2f}"  # Convert Tensors to scalars
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            




        cv2.imshow("Frame", frame)

        k = cv2.waitKey(10)
        if k == 27: 
            break
        