''' Recognizing faces & determining the presence of a mask in a real-time video '''

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

# Saving the known faces
face1 = face_recognition.load_image_file('Hero.jpg')
face1_enc = face_recognition.face_encodings(face1)[0]

known_encodings = [face1_enc]
known_names = ['Prantik']

# Function to recognize the faces & return the corresponding names
def recognizeFace(frame):
    
    rgb_frame = frame[:, :, ::-1] # Convert BGR to RGB
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
        
    return face_names


# Function to detect the face and determine the presence of mask 
def detectFace_and_predictMask(frame, faceDetector, maskDetector):

    # grabbing the dimensions of the frame and then constructing a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

	# passing the blob through the NeuralNet i.e. faceDetector and obtaining the face detections
	faceDetector.setInput(blob)
	detections = faceDetector.forward()
	print(detections.shape)

	# initializing our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = [] # A list of tuples
	preds = []

	# looping over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filtering out the weak detections by ensuring the confidence is
		# greater than a minimum confidence 
		if confidence > 0.5:
			# computing the (x, y)-coordinates of the bounding box for the detected face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extracting the face ROI, converting it from BGR to RGB channel
			# resizing it to 224x224, and preprocessing it for mobileNet
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# we make predictions if at least one face was detected
	if len(faces) > 0:

		faces = np.array(faces, dtype="float32")
		preds = maskDetector.predict(faces, batch_size=32)

	# return a tuple of the face locations and their corresponding mask detection
	return (locs, preds)


# loading the predefined faceDetector model
prototxtPath = r"E:/ML_projects/Face-Mask-Detection/face_detector/deploy.prototxt"
weightsPath = r"E:/ML_projects/Face-Mask-Detection/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceDetector = cv2.dnn.readNet(prototxtPath, weightsPath)

#loading the maskDetector model
maskDetector = load_model("E:/ML_projects/Face-Mask-Detection/SavedModel_2-0.99.h5")

''' Real Time Detection '''

vc = cv2.VideoCapture(0)

# looping over the frames from the video stream
while True:
    _, frame = vc.read(0)
    frame = imutils.resize(frame, width=600)
    
    (locations, predictions) = detectFace_and_predictMask(frame, faceDetector, maskDetector)
    names = recognizeFace(frame)
    
    for (box, pred, name) in zip(locations, predictions, names):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        acc = max(mask, withoutMask)*100
        label = f"{name} - {label} : {acc:.2f}%"
        
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # press 'Q' to stop the webcam feed
    if key == ord('q'):
        break
        
# Cleanup
cv2.destroyAllWindows()
vc.release()