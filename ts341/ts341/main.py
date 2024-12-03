from ultralytics import YOLO
import cv2
import math 

def main():
    # start webcam
    cap = cv2.VideoCapture("/home/louis/Documents/3A/Outils d'imagerie pour la robotique/TD/hornets/Contexte Projet Imagerie(1)/Rec_ruche_3_avec_frelon_5 Benoit Renaud_device_0_sensor_1_Color_0_image_data.mp4")
    cap.set(3, 640)
    cap.set(4, 480)

    # model
    model = YOLO("yolo-Weights/yolov8n.pt")

    # object classes
    classNames = ["insect", "frelon", "bee", "hornet", "guêpe", "frelon asiatique", "frelon européen", "abeille"
                ]


    while True:
        success, img = cap.read()
        results = model(img, stream=True, verbose=False)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()