##############################################################################
#
 import the necessary packages
from imutils.object_detection 
import non_max_suppression
from imutils
 import paths
import numpy as np

import argparse

import imutils

import cv2

import time

import pyttsx


###############################################################################

cascade_src = 'bananahaar.xml'

#video_src = 'video2.avi'

count = 0


#cap = cv2.VideoCapture(video_src)
 #Comment for real video 
capture
cap = cv2.VideoCapture('banana-96.bmp')

#cap = cv2.VideoCapture(0) 
#Uncomment for real time video
detector = cv2.CascadeClassifier(cascade_src)
 # Select the trained cascade classifier


################################################################################


while True:
    pedCount=0
    obCount= 0    
    ret, img = cap.read()
    origImg = img
    
   
 if (type(img) == type(None)):
       
 break
    
    
detector_cascade = cv2.CascadeClassifier(cascade_src) 
# Select the trained cascade classifier
    
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# Converts the image into grayScale image
    
objects = detector_cascade.detectMultiScale(gray, scaleFactor=1.3,
	minNeighbors=8, minSize=(75, 75))
 # Detect the number of cars in the video

    
    
# loop over the cat faces and draw a rectangle surrounding each
  
  for (i, (x, y, w, h)) in enumerate(objects):
           
 cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
           
 cv2.putText(img, "Object #{}".format(i + 1), (x, y - 10),
		
cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
          
  obCount= obCount+1
    
   
 cv2.imshow('Object_Recognized', img) 
# Display the frames

 
   if obCount >= 1:
       
 engine = pyttsx.init()
     
   Voice = "There are" +str(obCount)+ "Bananas"
        
engine.say(Voice)
       
 engine.runAndWait()
    
  
  ################################################################################################
   
 # initialize the HOG descriptor/person detector
  
  hog = cv2.HOGDescriptor()
   
 hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
  
  image = imutils.resize(origImg, width=min(400, origImg.shape[1]))
   
 # detect people in the image
   
 (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                padding=(8, 8), scale=1.05)
    
# draw the original bounding boxes
   
 for (x, y, w, h) in rects:
	  
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

  
  # apply non-maxima suppression to the bounding boxes using a
 
   # fairly large overlap threshold to try to maintain overlapping
   
 # boxes that are still people
   
 rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
   
 pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

   
 # draw the final bounding boxes
   
 for (xA, yA, xB, yB) in pick:
	   
 cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	  
  pedCount= pedCount+1
   
 # show the output images
  
  cv2.imshow("Pedestrians", image)

    
if pedCount >= 1:
       
 engine = pyttsx.init()
     
   #engine.say('One Pedestrian in Front')
      
  Voice = "and There are" +str(pedCount)+ "pedestrians in front"
       
 engine.say(Voice)
        
engine.runAndWait()
    

  
  if cv2.waitKey(33) == 27: #press 'ESC' key to stop the video
      
  break
  
  time.sleep(3)

cv2.destroyAllWindows()
