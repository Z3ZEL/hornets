import cv2
import numpy as np

DATASET_PATH = '../datasets/auto'


data_yaml = f"""
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: {DATASET_PATH} # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images
test: # test images (optional)

# Classes (80 COCO classes)
names:
    0: bee
    1: hornet
"""


def main(video_path):
    cap = cv2.VideoCapture(video_path)

    # params for corner detection 
    feature_params = dict(maxCorners=100, 
                          qualityLevel=0.3, 
                          minDistance=7, 
                          blockSize=7) 
    
    # Parameters for lucas kanade optical flow 
    lk_params = dict(winSize=(15, 15), 
                     maxLevel=2, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                               10, 0.03)) 
    
    # Create some random colors 
    color = np.random.randint(0, 255, (100, 3)) 
    
    
    frame_index = 0
    old_frame = None
    features_to_track = []
    

    while(1):
        k = cv2.waitKey(100) 
        cat = "train" if frame_index % 2 == 0 else "val"
        ret, frame = cap.read()
        display_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 4)

        ## Track features
        if len(features_to_track) > 0 and old_frame is not None:
            new_features_to_track = []
            for p0, class_name, old_bbox in features_to_track:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)
                ## If the majority of point is lost, discard the feature
                if np.sum(st) < 0.5 * p1.shape[0]:
                    continue


                st = st.reshape(p1.shape[0])
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                
                ## Recalculate the bounding box
                
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    a, b = int(a), int(b)
                    #Draw
                    cv2.circle(display_frame, (a, b), 5, color[i].tolist(), -1)
                try:
                    ## Push the new features to track
                    mass_center = np.mean(good_new, axis=0).reshape(2)
                    bbox = np.array([mass_center[0] - old_bbox[2]/2, mass_center[1] - old_bbox[3]/2, old_bbox[2], old_bbox[3]])
                    cv2.rectangle(display_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0,255,0), 2)
                except ValueError:
                    continue
                if len(good_new) > 0:
                    new_features_to_track.append((p1, class_name, bbox))    
                
                

            
            features_to_track = new_features_to_track
                    



        ## Check if key 'p' is pressed
        if k == ord('p'):
            bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

            mask = np.zeros_like(frame)
            mask[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])] = 255


            ## Get feature points in the selected region
            p0 = cv2.goodFeaturesToTrack(frame, mask = mask, **feature_params)
            
            ## Get only one feature point
            if p0 is not None:
                features_to_track.append((p0, "bee", bbox))

            # Draw features and bounding box
            for p in p0:
                a, b = p.ravel()
                a, b = int(a), int(b)
                cv2.circle(display_frame, (a, b), 5, (0, 0, 255), -1)
            cv2.rectangle(display_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0,255,0), 2)
            
            cv2.imshow("Frame", display_frame)
        
            key = cv2.waitKey(0)
            if key == ord('b') or key == ord('h'):
                ## Tag the selected region as bee or hornet
                features_to_track[-1] = (features_to_track[-1][0], "0" if key == ord('b') else "1", features_to_track[-1][2])
                
        ### Write to yolo txt file
        output_txt = ""

        for features in features_to_track:
            p0, class_name, bbox = features
            x, y, w, h = bbox
            x /= frame.shape[1]
            y /= frame.shape[0]
            w /= frame.shape[1]
            h /= frame.shape[0]
            output_txt += f"{class_name} {x} {y} {w} {h}\n"

        print(output_txt)






        if k == 27: 
            break
        frame_index += 1
        old_frame = frame.copy()
        # cv2.imwrite(f"{DATASET_PATH}/images/{cat}/frame{frame_index}.jpg", old_frame)

        cv2.imshow("Frame", display_frame)




    cv2.destroyAllWindows() 
    cap.release() 

if __name__ == "__main__":
    video_path = '../video/ruche01_frelon01_lores.mp4'  # Replace with your video file path
    # video_path = '../video/ruche03_frelon_lores.mp4'  # Replace with your video file path
    # video_path = '../video/filter_1_1_2.mp4'
    video_path = "/home/esrodriguez/Downloads/frame_video(1).mp4"
    main(video_path)