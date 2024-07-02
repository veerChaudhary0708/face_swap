import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


print('insightface',insightface.__version__)
print('numpy',np.__version__)

app=FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0,det_size=(640,640))

swapper=insightface.model_zoo.get_model('C:\DSA\inswapper_128.onnx',download=False,download_zip=False)

elisa_img=cv2.imread('C:\DSA\image (3).png')

elisa_faces=app.get(elisa_img)
elisa_face=elisa_faces[0]

import pyvirtualcam
from pyvirtualcam import PixelFormat


def func2(frame):
    # Convert BGR frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

     # Define red color range in HSV
    lower_red1 = (0, 60, 60)
    upper_red1 = (10, 255, 255)
    lower_red2 = (170, 50, 50)
    upper_red2 = (180, 255, 255)

    # Create masks for red color (two ranges due to hue wrapping in HSV)
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask = mask1 | mask2  # Combine masks using bitwise OR

    # Replace red pixels with blue (BGR format)
    frame[mask > 0] = (255, 0, 0)  # BGR value for blue

    return frame


def func(frame):
    faces3=app.get(frame)
    # processed_frame=frame.copy()
    for face3 in faces3:
         frame=swapper.get(frame, face3 ,elisa_face ,paste_back=True)
    return frame

def modify_video_from_camera(func):
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Read the first frame to get the frame dimensions
    ret, frame = cap.read()
    # if not ret:
    #     print("Error: Could not read frame")
    #     cap.release()
    #     return

    # Get frame dimensions
    height, width, _ = frame.shape
    # Define reduced dimensions
    # reduced_width = width // 2
    # reduced_height = height // 2

    # Create the virtual camera
    # with pyvirtualcam.Camera(reduced_width, reduced_height, fps=10, fmt=PixelFormat.BGR) as cam:
    with pyvirtualcam.Camera(width, height, fps=10, fmt=PixelFormat.BGR) as cam:
        # print(f'Using virtual camera: {cam.device}')

        while True:
            ret, frame = cap.read()

            # if not ret:
            #     print("No more frames to read")
            #     break

            # Apply the function to the frame
            modified_frame = func(frame)

            # Send the modified frame to the virtual camera
            cam.send(modified_frame)
            # cam.sleep_until_next_frame()

            # Optionally display the modified frame in a window (for debugging)
            # cv2.imshow('Modified Video', modified_frame)

            # Break the loop if 'q' is pressed
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    # Release resources
    # cap.release()
    # cv2.destroyAllWindows()

# Example function to apply to each frame


# Example usage
modify_video_from_camera(func)




