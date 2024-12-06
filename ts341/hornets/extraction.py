import cv2
import numpy as np
import pyrealsense2 as rs
import sys
import os


RESOLUTION_FACTOR = 0.5
MAX_RANGE = 2
FRAMERATE = 30
MIN_RANGE = 0.2

def check_path(file, output_folder):
    if not file:
        print("Please provide a bag file")
        sys.exit(1)

    if not output_folder:
        print("Please provide an output folder")
        sys.exit(1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


if len(sys.argv) < 3:
    print("Not enough arguments --  poetry run extract bag_file output_folder ")
    sys.exit(1)
check_path(sys.argv[1], sys.argv[2])

def main_video(bag_file=sys.argv[1], output_folder=sys.argv[2]):
    ## Convert bag file to video
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    print("Starting pipeline")
    profile = pipe.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align_to = rs.stream.color
    align = rs.align(align_to)
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise Exception("No color frames found in the bag file.")
    frame_width = color_frame.get_width()
    frame_height = color_frame.get_height()
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(f'{output_folder}/output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (int(frame_width*RESOLUTION_FACTOR), int(frame_height*RESOLUTION_FACTOR)))
    print("Writing video")
    while True:
        try:
            frames = pipe.wait_for_frames(timeout_ms=1000)
        except:
            break
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            break

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        color_image = cv2.resize(color_image, (int(frame_width*RESOLUTION_FACTOR), int(frame_height*RESOLUTION_FACTOR)))

        out.write(color_image)


    out.release()
    pipe.stop()


def main(bag_file=sys.argv[1], output_folder=  sys.argv[2]):
    divide = int(sys.argv[3]) if len(sys.argv) >= 4 else 1

    pipe = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    profile = pipe.start(config)


    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    


    align_to = rs.stream.depth
    align = rs.align(align_to)


    key = " "
    skip = 0
    while not key == ord("q"):
        skip += 1
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)


        color_frame = frames.get_color_frame()
        color_frame = np.asarray(color_frame.get_data())
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)


        depth_frame = aligned_frames.get_depth_frame()
        depth_frame = np.asanyarray(depth_frame.get_data())


        # TODO : Use max_range and min_range to get the best resolution
        depth_uint8 = np.uint8(depth_frame * 255)
        depth_float = np.float32(depth_frame) * depth_scale

        #Get images size
        depth_height, depth_width = depth_frame.shape
        color_height, color_width, _ = color_frame.shape



        #Resize
        color_frame = cv2.resize(color_frame, (int(color_width*RESOLUTION_FACTOR), int(color_height*RESOLUTION_FACTOR)))  
        depth_uint8 = cv2.resize(depth_uint8, (int(depth_width*RESOLUTION_FACTOR), int(depth_height*RESOLUTION_FACTOR)))


        if skip % divide == 0:
            cv2.imshow("Depth", depth_uint8)
            cv2.imwrite(output_folder + f"/depth_frame{skip}.jpg", depth_uint8)
            cv2.imshow("Hornets", color_frame)
            cv2.imwrite(output_folder + f"/color_frame{skip}.jpg", color_frame)
        key = cv2.waitKey(1)



def extract_mp4(mp4_file=sys.argv[1], output_folder=  sys.argv[2]):
    divide = int(sys.argv[3]) if len(sys.argv) >= 4 else 1

    range_txt = sys.argv[4] if len(sys.argv) >= 5 else ""

    ### Parse range like "10:20,50:52 ..."
    ranges = []
    if range_txt:
        for r in range_txt.split(","):
            start, end = r.split(":")
            ranges.append((int(start), int(end)))
    print("Ranges", ranges)
    

    cap = cv2.VideoCapture(mp4_file)

    
    if not cap.isOpened():
        print("Error opening video stream or file")
        sys.exit(1)

    frame_count = 0
    print("Divide", divide)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % divide == 0:
            # print("Saving frame", frame_count)
            if len(ranges) > 0:
                is_in_range = False
                current_seconds = frame_count / FRAMERATE
                
                for start, end in ranges:
                    if current_seconds >= start and current_seconds <= end:
                        is_in_range = True
                        print("Current seconds", current_seconds)
                        break
                else:
                    continue

                if not is_in_range:
                    continue

            # frame_resized = cv2.resize(frame, (int(frame.shape[1] * RESOLUTION_FACTOR), int(frame.shape[0] * RESOLUTION_FACTOR)))
            frame_resized = cv2.resize(frame, (640, 640))
            cv2.imshow('Frame', frame_resized)
            cv2.imwrite(output_folder + f"/frame{frame_count}.jpg", frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()