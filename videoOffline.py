import os
import cv2
import time
import numpy as np
import argparse

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video file")
ap.add_argument("-s", "--save", required=True, help="path to save detected video")
ap.add_argument("--fps", type=int, default=0, help="fps to detect video")
ap.add_argument("--model", required=True, help="loading model to detect video")
ap.add_argument("--weight", required=True, help="weight to detect video")
args = vars(ap.parse_args())

#load model
model = model_from_json(open(args["model"],"r").read())
#load weights
model.load_weights(args["weight"])

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# input video
cap = cv2.VideoCapture(args["video"])
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fpsCap = int(cap.get(cv2.CAP_PROP_FPS))

# arg fps
fpsReduce = args["fps"]

# flg to check apply fps
fpsFlg = True
if fpsReduce <= 0 or fpsReduce >= fpsCap:
    fpsReduce = fpsCap
    fpsFlg = False

# output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args["save"], fourcc, fpsCap, (width,height))


# used to record emotions
result = [0, 0, 0, 0, 0, 0, 0]

# flg to get detect frame
flg = True

while cap.isOpened():
    # captures frame and returns boolean value and captured image
    ret,test_img=cap.read()
    # get frame number
    currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if fpsFlg:
        if flg:
            firstFrame = currentFrame
            lastFrame = currentFrame + fpsCap

        if currentFrame == lastFrame:
            flg = True

    if (fpsFlg == False) or (currentFrame >= firstFrame and currentFrame <= firstFrame + fpsReduce):
        flg = False
        if not ret:
            continue
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray=gray_img[y:y+w,x:x+h] #cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            #find max indexed array
            max_index = np.argmax(predictions[0])
            if max_index >= 0:
                result[max_index] += 1

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)   
        resized_img = cv2.resize(test_img, (width, height))
        cv2.imshow('Facial emotion analysis ', resized_img)
        out.write(resized_img)
    
    if(currentFrame == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        break

print(*result, sep = ",")
cap.release()
out.release()
cv2.destroyAllWindows

