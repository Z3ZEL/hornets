import cv2
import numpy as np
import pyrealsense2 as rs
import sys
import os


RESOLUTION = (640, 480)


def check_path(file, output_folder):
    if not file:
        print("Please provide a bag file")
        sys.exit(1)

    if not output_folder:
        print("Please provide an output folder")
        sys.exit(1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


max_range = 2
min_range = 0.2

def main_video(bag_file=sys.argv[1], output_folder=  sys.argv[2]):
    check_path(bag_file, output_folder)
    ## Convert bag file to video
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    profile = pipe.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align_to = rs.stream.color
    align = rs.align(align_to)

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, RESOLUTION)

    while True:
        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            break

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        out.write(color_image)

    out.release()
    pipe.stop()



def main(bag_file=sys.argv[1], output_folder=  sys.argv[2], divide = sys.argv[3]):
    divide = divide if divide is not None else 1
    check_path(bag_file, output_folder)

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


        color_frame = aligned_frames.get_color_frame()
        color_frame = np.asarray(color_frame.get_data())
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)


        depth_frame = aligned_frames.get_depth_frame()
        depth_frame = np.asanyarray(depth_frame.get_data())


        # TODO : Use max_range and min_range to get the best resolution
        depth_uint8 = np.uint8(depth_frame * 255)
        depth_float = np.float32(depth_frame) * depth_scale

        if skip % divide == 0:
            cv2.imshow("Depth", depth_uint8)
            cv2.imwrite(output_folder + "/depth_frame.jpg", depth_uint8)
            cv2.imshow("Hornets", color_frame)
            cv2.imwrite(output_folder + "/color_frame.jpg", color_frame)
        key = cv2.waitKey(1)