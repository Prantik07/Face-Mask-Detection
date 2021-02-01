''' Using the face locations provided by face recog function '''

# Importing the required packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import face_recognition
import numpy as np
import imutils
import time
import cv2
import os


face1 = face_recognition.load_image_file('Hero.jpg')
face1_enc = face_recognition.face_encodings(face1)[0]

known_encodings = [face1_enc]
known_names = ['Prantik']


def detectFace_and_predictMask(frame, maskDetector):
    
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        name = "Unknown"
        match = face_recognition.compare_faces(known_encodings, face_encoding)
        distance = face_recognition.face_distance(known_encodings, face_encoding)
        least_distance_index = np.argmin(distance)
        if match[least_distance_index]:
            name = known_names[least_distance_index]
        face_names.append(name)
        
    faces = [] 
    locs = [] #coordinates of the face locations in the required format
    for (startY, endX, endY, startX) in face_locations:
        face = frame[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(frame, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        
        faces.append(face)
        locs.append((startX, startY, endX, endY))
        
    preds = []
    if len(faces)>0:
        faces = np.array(faces, dtype = "float32")
        preds = maskDetector.predict(faces, batch_size=32)
        
    return (face_names, locs, preds)

#loading the maskDetector model
maskDetector = load_model("E:/Face-Mask-Detection-master/SavedModel_2.h5")

''' Real Time Detection '''

vc = cv2.VideoCapture(0)

# looping over the frames from the video stream
while True:
    _, frame = vc.read(0)
    frame = imutils.resize(frame, width=400)
    
    (names, locations, predictions) = detectFace_and_predictMask(frame, maskDetector)
    
    for (name, box, pred) in zip(names, locations, predictions):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        cat = max(mask, withoutMask)
        label = f"{name} - {label} : {cat:.2f}%"
        
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
        

# do a bit of cleanup
cv2.destroyAllWindows()
vc.release()
