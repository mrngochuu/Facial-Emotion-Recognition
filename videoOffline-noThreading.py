import os
import cv2
import time
import numpy as np
import argparse
import imutils
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from imutils.video import FPS


#load model
model = model_from_json(open("fer.json","r").read())
#load weights
model.load_weights('fer.h5')

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# cap=cv2.VideoCapture('video_offline/test.mp4')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
args = vars(ap.parse_args())

# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
stream = cv2.VideoCapture(args["video"])
# start the FPS timer
fps = FPS().start()

# used to record emotions
i = 0
result = [0, 0, 0, 0, 0, 0, 0]

while True:
    
    
    (grabbed, frame) = stream.read()
    # ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    
    
    textEmotion = ''
    if not grabbed:
	    break
    # if not frame:
    #     continue
    gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
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

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        textEmotion = predicted_emotion
        
   
    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)
    print('--------------------------------')
    print(str(i) + ': ' + textEmotion)
    print(*result, sep = ", ")
    print('--------------------------------')
    i += 1

    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break
    fps.update()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
stream.stop()

# cap.release()
# cv2.destroyAllWindows