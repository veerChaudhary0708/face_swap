import cv2
import numpy as np
import pyvirtualcam

# Initialize camera capture

def func2(frame):
     # Define lower and upper BGR limits for red color
    lower_red = (0, 0, 50)
    upper_red = (50, 50, 255)

    # Create mask for red color in BGR space
    mask = cv2.inRange(frame, lower_red, upper_red)

    # Replace red pixels with blue color (BGR format)
    frame[mask > 0] = (0, 255, 0)  # BGR value for blue
    return frame

def func3(frame):
    frame = cv2.bitwise_not(frame)
    return frame
cap = cv2.VideoCapture(0)  # 0 for default camera, change if necessary

# Create a virtual camera
with pyvirtualcam.Camera(width=640, height=480, fps=30) as virtual_cam:
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            break
        
        
        # modified_frame = cv2.bitwise_not(frame)
        modified_frame=func3(frame)
        virtual_cam.send(modified_frame)
        
        # cv2.imshow('Modified Camera Feed', modified_frame)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()


