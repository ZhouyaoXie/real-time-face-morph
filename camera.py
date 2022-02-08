import cv2 as cv
import numpy as np
import os

 
# Open the webcam
cam = cv.VideoCapture(0)

print(os.getcwd(), os.listdir())

 
modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
cvNet = cv.dnn.readNetFromTensorflow(modelFile, configFile)

transform_model = cv.dnn.readNetFromTorch()

# pb  = 'frozen_inference_graph.pb'
# pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'
 
# # Read the neural network
# cvNet = cv.dnn.readNetFromTensorflow(pb,pbt)   
 
while True:
 
  # Read in the frame
  ret_val, img = cam.read()
  rows = img.shape[0]
  cols = img.shape[1]
  cvNet.setInput(cv.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False))
 
  # Run object detection
  cvOut = cvNet.forward()
 
  # Go through each object detected and label it
  for i in range(cvOut.shape[2]):
    confidence = cvOut[0, 0, i, 2]
    if confidence > .3:
      left = int(cvOut[0,0,i,3] * cols)
      top = int(cvOut[0,0,i,4] * rows)
      right = int(cvOut[0,0,i,5] * cols)
      bottom = int(cvOut[0,0,i,6] * rows)
      cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

 
  # Display the frame
  cv.imshow('my webcam', img)
 
  # Press ESC to quit
  if cv.waitKey(1) == 27: 
    break
 
# Stop filming
cam.release()
 
# Close down OpenCV
cv.destroyAllWindows()