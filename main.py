# Jonathan Kleppinger 
# Motion detection tigger for facial recognition
  
# importing OpenCV
import cv2
# Facial Recog
import face_recognition
# importing datetime class from datetime library 
import numpy as np
from datetime import datetime 

## Initialize for motion detection
# Assigning our static_back to None 
static_back = None
# List when any moving object appear 
motion_list = [ None, None ] 
#trigger
trigger = 0
# Loop counter
loop = 0

## Initialize for facial recognition
# Load a sample picture and learn how to recognize it.

user_image1 = face_recognition.load_image_file("jonny.jpeg")
user_face_encoding1 = face_recognition.face_encodings(user_image1)[0]

user_image2 = face_recognition.load_image_file("chantelle.jpg")
user_face_encoding2 = face_recognition.face_encodings(user_image2)[0]

user_image3 = face_recognition.load_image_file("brett.jpg")
user_face_encoding3 = face_recognition.face_encodings(user_image3)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    user_face_encoding1,
    user_face_encoding2,
    user_face_encoding3,


]
known_face_names = [
    "Jonathan Kleppinger",
    "Chantelle Hobbs",
    "Bread"    
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
tempNames = [] # Stores the names of the faces recognized


# Capturing video 
video = cv2.VideoCapture(0) 
  
# Infinite while loop to treat stack of image as video 
while True: 
    tempNames = [] # Reset the names detected when only motion detecting.

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
        trigger += 1
        print('why')
    print(trigger)

    # Displaying the black and white image in which if 
    # intensity difference greater than 30 it will appear white 
    cv2.imshow("Threshold Frame", thresh_frame) 
  
    # Displaying color frame with contour of motion of object 
    cv2.imshow("Color Frame", frame) 

    loop += 1
    if loop > 500:
        static_back = None
        loop = 0
  
    # if q entered whole process will stop 
    # Hit 'q' on the keyboard to quit!
    if (cv2.waitKey(1) & 0xFF == ord('q')) | (trigger > 10):
        break
  

while trigger > 10:
    # Grab a single frame of video
    ret, frame = video.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                nameFound = False;
                if len(tempNames) == 0:  # If the list of names detected is empty
                    tempNames += [name];
                else:
                    for i in range(0, len(tempNames)):  # Checks to see if the list contains a detected name
                        if(tempNames[i] == name):
                            nameFound = True;
                            break;

                if(nameFound == False):     # If the name was not previously detected
                    with open('facesDetected.txt', 'a') as f:
                        now = datetime.now() # current date and time
                        currTime = now.strftime("%d/%m/%Y %H:%M:%S")
                        f.write(name + " " + currTime)
                        f.write('\n')

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

## END CODE
video.release()
# Destroying all the windows 
cv2.destroyAllWindows()
