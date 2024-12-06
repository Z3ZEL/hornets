import cv2
import numpy as np
import pyrealsense2 as rs


max_range = 2
min_range = 0.2


def main(bag_file):

    pipe = rs.pipeline()
    config = rs.config()

    config.enable_device_from_file(bag_file, repeat_playback=False)

    profile = pipe.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.depth
    align = rs.align(align_to)

    key = " "
    while not key == ord("q"):
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

        cv2.imshow("Hornets", color_frame)
        key = cv2.waitKey(1)


if __name__ == "__main__":
    bag_file = "../vid/ruche01_frelon01.bag"
    main(bag_file)
