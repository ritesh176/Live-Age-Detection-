# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 17:40:09 2023

@author: rites
"""

import cv2
import math
import argparse


# Load the pre-trained models
ageProto =r"C:\Users\rites\Downloads\age_deploy.prototxt"
ageModel = r"C:\Users\rites\Downloads\age_net.caffemodel"
genderProto =r"C:\Users\rites\Downloads\gender_deploy.prototxt"
genderModel =r"C:\Users\rites\Downloads\gender_net.caffemodel"

# Load the models using OpenCV's dnn module
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# List of age and gender labels
ageLabels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderLabels = ['Male', 'Female']

# Load the input image
image = cv2.imread(r"C:\Users\rites\Downloads\happy image.4.jpg")

# Preprocess the image
blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False, crop=False)

# Set the input to the age and gender networks
ageNet.setInput(blob)
genderNet.setInput(blob)

# Perform age and gender detection
agePreds = ageNet.forward()
genderPreds = genderNet.forward()

# Get the predicted age and gender
ageIdx = agePreds[0].argmax()
genderIdx = genderPreds[0].argmax()
age = ageLabels[ageIdx]
gender = genderLabels[genderIdx]

# Display the age and gender
text = f"Age: {age}, Gender: {gender}"
cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Display the image
cv2.imshow("Age and Gender Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
