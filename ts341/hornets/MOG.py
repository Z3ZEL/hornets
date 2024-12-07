import numpy as np
import cv2 as cv


def otsu_thresholding(img):
    """Apply OTSU thresholding to the image"""
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    
    img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)    

    img = cv.equalizeHist(img)
    
    _, th = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return th


def otsu_blur_thresholding(img):
    """Apply OTSU thresholding to the image after blurring it"""
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    
    img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)    

    img = cv.equalizeHist(img)
    blur = cv.GaussianBlur(img, (5, 5), 0)
    _, th = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return th


def mean_adaptive_thresholding(img):
    """Apply mean adaptive thresholding to the image"""
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
    
    img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)    

    img = cv.equalizeHist(img)
    
    th = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2
    )
    return th


def gaussian_adaptive_thresholding(img):
    """Apply gaussian adaptive thresholding to the image"""
    
    # brightness = 1 
    # # Adjusts the contrast by scaling the pixel values by 2.3 
    # contrast = 1
    # img = cv.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)
    
    # kernel = np.ones((10, 10), np.uint8)
    
    # img = cv.erode(img, kernel, iterations=1)  
    # img = cv.dilate(img, kernel, iterations=1)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)    

    img = cv.equalizeHist(img)
    
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 8)
    
    return img


def basic():
    
    cap = cv.VideoCapture('/home/louis/Documents/3A/Outils_imagerie/TD/hornets/ts341/videos/Rec_ruche_3_avec_frelon_5 Benoit Renaud_device_0_sensor_1_Color_0_image_data.mp4')
    # cap = cv.VideoCapture('/home/louis/Documents/3A/Outils_imagerie/TD/hornets/video/ruche01_frelon01_lores.mp4')

    fgbg = cv.createBackgroundSubtractorMOG2()
        
    while(1):
        ret, frame = cap.read()
        
        frame = cv.resize(frame, (640, 360)) 
        
        cv.imshow("frame", frame)
        
        fgmask = gaussian_adaptive_thresholding(frame)
        
        fgmask = fgbg.apply(fgmask)
        
        # cv.imshow('fgmask',fgmask)
        
        # n=3
        # kernel = np.ones((n, n), np.uint8)
        # #fgmask = cv.erode(fgmask, kernel, iterations=1)  
        # fgmask = cv.dilate(fgmask, kernel, iterations=1)
        
        cv.imshow('fgmask_k', fgmask)
        if cv.waitKey(30) & 0xFF == ord("q"):
                break

    cap.release()
    cv.destroyAllWindows()

    
def basic_2():
    
    cap = cv.VideoCapture('/home/egouhey/Documents/3A/outil_imagerie/data_hornet/Contexte Projet Imagerie(1)/Rec_ruche_3_avec_frelon_5 Benoit Renaud_device_0_sensor_1_Color_0_image_data.mp4')
    # cap = cv.VideoCapture('/home/egouhey/Documents/3A/outil_imagerie/hornets/video/ruche01_frelon01_lores.mp4')

    fgbg = cv.createBackgroundSubtractorMOG2()
    
    while(1):
        ret, frame = cap.read()
        
        frame = cv.resize(frame, (640, 360)) 
        
        cv.imshow("frame", frame)
        
        frame_gauss = gaussian_adaptive_thresholding(frame)
        
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        th_h, th_s, th_v = cv.split(hsv)
        
        # cv.imshow("th_h.png",th_h)
        # cv.imshow("th_s.png",th_s)
        # cv.imshow("th_v.png",th_v)
        
        # fgmask_h= fgbg.apply(th_h)
        # cv.imshow('fgmask_h',fgmask_h)
        
        n = 2
        kernel = np.ones((n, n), np.uint8)
        th_s = cv.erode(th_s, kernel, iterations=1)  
        th_s = cv.dilate(th_s, kernel, iterations=1)
        
        fgmask_s = fgbg.apply(th_s)
        cv.imshow('fgmask_s', fgmask_s)  
        
        # fgmask_v= fgbg.apply(th_v)
        # cv.imshow('fgmask_v',fgmask_v)
        
        # final= fgbg.apply(th_s)
        # cv.imshow('final',final) 
          
        if cv.waitKey(30) & 0xFF == ord("q"):
                break

    cap.release()
    cv.destroyAllWindows()

def filtre_2(frame, fgbg):
    kernel_flou = np.array([[1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1]]) / 16 

    kernel_gaussian = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]) / 16

    kernel_int = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]]) 

    kernel_laplacian = np.array([[ 0, 1, 0],
                        [ 1, -4, 1],
                        [ 0, 1, 0]])
    
    
    output = cv.filter2D(frame, -1, kernel_flou)
    
    
    output = fgbg.apply(output)
    
    frame_masked = cv.bitwise_and(frame, frame, mask=output)

    return frame_masked


    # saving the video file
    # out.write(cv.cvtColor(frame_masked, cv.COLOR_GRAY2BGR))
def basic_3():
    
    cap = cv.VideoCapture('/home/louis/Documents/3A/Outils_imagerie/TD/hornets/video/ruche03_frelon_lores.mp4')
    # cap = cv.VideoCapture('/home/egouhey/Documents/3A/outil_imagerie/hornets/video/ruche01_frelon01_lores.mp4')

    fgbg = cv.createBackgroundSubtractorMOG2()
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    output_video = "output_recorded.mp4"

    # create the `VideoWriter()` object
    out = cv.VideoWriter(output_video, fourcc, 30, frame_size)

    if not cap.isOpened():
        print("Error opening video file")
    count = 0
    # Read until video is completed
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:

            # frame = cv.resize(frame, (640, 360)) 
            
            cv.imshow("frame", frame)
            
            kernel_flou = np.array([[1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1]]) / 16 

            kernel_gaussian = np.array([[1, 2, 1],
                                        [2, 4, 2],
                                        [1, 2, 1]]) / 16

            kernel_int = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]]) 

            kernel_laplacian = np.array([[ 0, 1, 0],
                                [ 1, -4, 1],
                                [ 0, 1, 0]])
            
            # output = cv.filter2D(frame, -1, kernel_laplacian)        
            # output = cv.filter2D(output, -1, kernel_gaussian)
            output = cv.filter2D(frame, -1, kernel_flou)
            cv.imshow("f", output)
            
            output = fgbg.apply(output)

            # Find contours
            # contours, hierarchy = cv.findContours(output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # min_contour_area = 500  # Define your minimum area threshold
            # # filtering contours using list comprehension
            # large_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]
            # frame_out = output.copy()
            # for cnt in large_contours:
            #     x, y, w, h = cv.boundingRect(cnt)
            #     frame_out = cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 200), 3)
            
            frame_masked = cv.bitwise_and(frame, frame, mask=output)


            # saving the video file
            # out.write(cv.cvtColor(frame_masked, cv.COLOR_GRAY2BGR))
            out.write(frame_masked)

            cv.imshow("filter2D", frame_masked)

            if cv.waitKey(30) & 0xFF == ord("q"):
                break
        else:
            cap.release()
            out.release()

    cv.destroyAllWindows()
  
    
def flow():
        
    capture = cv.VideoCapture('/home/egouhey/Documents/3A/outil_imagerie/data_hornet/Contexte Projet Imagerie(1)/Rec_ruche_3_avec_frelon_5 Benoit Renaud_device_0_sensor_1_Color_0_image_data.mp4')
    # cap = cv.VideoCapture('/home/egouhey/Documents/3A/outil_imagerie/hornets/video/ruche01_frelon01_lores.mp4')

    # Reading the first frame
    _, frame1 = capture.read()
    frame1 = gaussian_adaptive_thresholding(frame1)
    frame1 = cv.resize(frame1, (640, 360)) 
    # Convert to gray scale
    # prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    prvs = frame1
    # Create mask
    hsv_mask = np.zeros_like(frame1)
    # Make image saturation to a maximum value
    hsv_mask[..., 1] = 255
    
    # Till you scan the video
    while(1):
        
        # Capture another frame and convert to gray scale
        _, frame2 = capture.read()
        frame2 = gaussian_adaptive_thresholding(frame2)
        frame2 = cv.resize(frame2, (640, 360)) 
        cv.imshow('origin', frame2)
        # next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        next = frame2
    
        # Optical flow is now calculated
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute magnite and angle of 2D vector
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # Set image hue value according to the angle of optical flow
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        # Set value as per the normalized magnitude of optical flow
        hsv_mask[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        # Convert to rgb
        rgb_representation = cv.cvtColor(hsv_mask, cv.COLOR_HSV2BGR)
    
        cv.imshow('frame2', rgb_representation)
        kk = cv.waitKey(20) & 0xff
        prvs = next
    
    capture.release()

    cv.destroyAllWindows()

    
def generic():
    cap = cv.VideoCapture('/home/egouhey/Documents/3A/outil_imagerie/data_hornet/Contexte Projet Imagerie(1)/Rec_ruche_3_avec_frelon_5 Benoit Renaud_device_0_sensor_1_Color_0_image_data.mp4')
    # cap = cv.VideoCapture('/home/egouhey/Documents/3A/outil_imagerie/hornets/video/ruche01_frelon01_lores.mp4')

    fgbg = cv.createBackgroundSubtractorMOG2()
    
    # Define the codec and create two Videor objects
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Use 'MJPG' or 'XVID' codec
    width = int(cap.get(3)) 
    height = int(cap.get(4)) 
    # Video Writer for the processed frames
    # out_frame = cv.VideoWriter('frame_video.mp4', fourcc, 30, (width, height))  # 30 fps
    # out_fgmask = cv.VideoWriter('fgmask_video.mp4', fourcc, 30, (width, height))  # 30 fps
    
    while(1):
        ret, frame = cap.read()
        frame = cv.resize(frame, (640, 360)) 
        
        cv.imshow("frame", frame)
        
        brightness = 1
        # # Adjusts the contrast by scaling the pixel values by 2.3 
        contrast = 2
        frame = cv.addWeighted(frame, contrast, np.zeros(frame.shape, frame.dtype), 0, brightness)
        
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        th_h, th_s, th_v = cv.split(hsv)
        
        # ret_h, th_h = cv.threshold(h,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        # ret_s, th_s = cv.threshold(s,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        # ret_v, th_v = cv.threshold(v,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        
        # cv.imshow("th_h.png",th_h)
        # cv.imshow("th_s.png",th_s)
        # cv.imshow("th_v.png",th_v)
        
        frame = gaussian_adaptive_thresholding(frame)

        # #frame_1=cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        # # out_frame.write(frame_1).write
        
        # cv.imshow('frame',frame)  
        # fgmask_h = fgbg.apply(th_h)
        
        # fgmask_s = fgbg.apply(th_s)
        
        # fgmask_v = fgbg.apply(th_v)
        
        fgmask_and = cv.bitwise_and(cv.bitwise_xor(th_h, th_s) , th_s) 
        
        # fgmask_and = cv.normalize(fgmask_and, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)    

        # fgmask_and = cv.equalizeHist(fgmask_and)
        
        # fgmask_and = fgbg.apply(fgmask_and)
            
        kernel = np.ones((2, 2), np.uint8)
        
        # img = cv.erode(img, kernel, iterations=1)  
        fgmask_and = cv.dilate(fgmask_and, kernel, iterations=1)
        
        cv.imshow('fgmask', fgmask_and)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    
    # out_frame.release()
    # out_fgmask.release()
    
    cap.release()
    cv.destroyAllWindows()

    
def main():
    basic_3()
