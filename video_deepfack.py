import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app=FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0,det_size=(640,640))

swapper=insightface.model_zoo.get_model('C:\DSA\inswapper_128.onnx',download=False,download_zip=False)

veer_img=cv2.imread('C:\DSA\image (3).png')
veer_faces=app.get(veer_img)
veer_face=veer_faces[0]


#for image
# img=cv2.imread('/home/balveerguleriya2131836/Untitled design (3).png')
# faces3=app.get(img)
# for face3 in faces3:
#          img=swapper.get(img, face3 ,veer_face ,paste_back=True)
# plt.imshow(img[:,:,::-1])
# plt.show()         

#for video
def func(frame):
    faces3=app.get(frame)
    processed_frame=frame.copy()
    # Example processing: convert the frame to grayscale
    #processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for face3 in faces3:
         processed_frame=swapper.get(processed_frame, face3 ,veer_face ,paste_back=True)
    return processed_frame

def modify_video(video_path, output_path, func):
  """Modifies each frame of a video using the provided function.

  Args:
      video_path (str): Path to the input video file.
      output_path (str): Path to save the modified video file.
      func (callable): Function to apply to each frame.
  """

  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    print("Error opening video stream or file")
    return

  # Get video properties for output video
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)

  # Create a video writer with the same properties as the input video
  fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Adjust fourcc for different codecs if needed
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  while True:
    ret, frame = cap.read()

    if not ret:
      print("No more frames to read")
      break

    # Apply the function to the frame
    modified_frame = func(frame)

    # Write the modified frame to the output video
    out.write(modified_frame)

  # Release resources
  cap.release()
  out.release()
  #cv2.destroyAllWindows()
  print("Video modification complete!")


video_path = "C:\DSA\.cph\WIN_20240624_11_16_52_Pro.mp4"
output_path = "C:\DSA\output3_vid.avi"  # Adjust extension based on desired codec

modify_video(video_path, output_path, func)