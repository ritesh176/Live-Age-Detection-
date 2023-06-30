# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:10:27 2023

@author: rites
"""


#pip install --upgrade opencv-python
import math
import cv2
import argparse

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes
ageModel = r"C:\Users\rites\Downloads\age_net.caffemodel"
ageProto = r"C:\Users\rites\Downloads\age_deploy.prototxt"


genderProto =r"C:\Users\rites\Downloads\gender_deploy.prototxt"
genderModel =r"C:\Users\rites\Downloads\gender_net.caffemodel"
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
genderList = ['Male', 'Female']

# Assuming 'face' is obtained from face detection
# Load the input image
image = cv2.imread(r"C:\Users\rites\Downloads\happy image.4.jpg")

# Perform face detection to obtain the face region
face_cascade = cv2.CascadeClassifier("path/to/haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
(x, y, w, h) = faces[0]
face = image[y:y+h, x:x+w]

# Preprocess the face region
blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

genderNet.setInput(blob)
genderPreds = genderNet.forward()
gender = genderList[genderPreds[0].argmax()]
print("Gender Output : {}".format(genderPreds))
print("Gender : {}".format(gender))


label = "{}, {}".format(gender, age)
cv2.putText(frameFace, label, (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3, cv2.LINE_AA)
cv2.imshow("Age Gender Demo", frameFace)