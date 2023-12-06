# Jonathan Kleppinger 
# Python program to implement  
# Webcam Motion Detector 
  
# importing OpenCV, time and Pandas library 
import cv2, time
# importing datetime class from datetime library 
from datetime import datetime 
import numpy as np

# Assigning our static_back to None 
static_back = None
  
# List when any moving object appear 
motion_list = [ None, None ] 
  
# Capturing video 
video = cv2.VideoCapture(0) 
  
#trigger
trigger = 0

# Infinite while loop to treat stack of image as video 
while True: 
    # Reading frame(image) from video 
    check, frame = video.read() 
  
    # Initializing motion = 0(no motion) 
    motion = 0
  
    # Converting color image to gray_scale image 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    # Converting gray scale image to GaussianBlur  
    # so that change can be find easily 
    gray = cv2.GaussianBlur(gray, (21, 21), 0) 
  
    # In first iteration we assign the value  
    # of static_back to our first frame 
    if static_back is None:
        static_back = gray 
        continue
  
    # Difference between static background 
    # and current frame(which is GaussianBlur) 
    diff_frame = cv2.absdiff(static_back, gray) 
  
    # If change in between static background and 
    # current frame is greater than 30 it will show white color(255) 
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 

    # trigger that shows motion has been detected 
    bright_count = np.sum(np.array(thresh_frame)>=1)      
    print(int(bright_count))
    if bright_count > 100000:
        tigger = 1

    # Displaying image in gray_scale 
    cv2.imshow("Gray Frame", gray) 
  
    # Displaying the difference in currentframe to 
    # the staticframe(very first_frame) 
    cv2.imshow("Difference Frame", diff_frame) 
  
    # Displaying the black and white image in which if 
    # intensity difference greater than 30 it will appear white 
    cv2.imshow("Threshold Frame", thresh_frame) 
  
    # Displaying color frame with contour of motion of object 
    cv2.imshow("Color Frame", frame) 
  
    key = cv2.waitKey(1) 
    # if q entered whole process will stop 
    if key == ord('q'): 
        # if something is movingthen it append the end time of movement 
        if motion == 1: 
            time.append(datetime.now()) 
        break
  

video.release() 
  
# Destroying all the windows 
cv2.destroyAllWindows()