import os
import cv2
import numpy as np
import tensorflow as tf
import mtcnn
import argparse
from keras.models import load_model
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
from skimage import color


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video file")
ap.add_argument("-s", "--save", required=True, help="path to save detected video")
ap.add_argument("-f", "--fps", type=int, default=10, help="fps to detect video")
ap.add_argument("-w", "--weight", required=True, help="weight to detect video")
ap.add_argument("-c", "--confidence", required=True, default=0.0, help="Confidence of emotions on videos")
args = vars(ap.parse_args())

model = load_model(args["weight"])

# use CPU
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# etract face
def extract_face(face,image,required_size=(224,224)):
  x1, y1, width, height = face['box']
  x2, y2 = x1 + width, y1 + height
  if (x1 < 0):
    x1 = 0
  if (x2 < 0):
    x2 = 0
  if (y1 < 0):
    y1 = 0
  if (y2 < 0):
    y2 = 0
  
  face_boundary = image[y1:y2, x1:x2]
  face_image = Image.fromarray(face_boundary)
  face_image = face_image.resize(required_size)
  face_array = asarray(face_image)
  return face_array

# change gray img
def rgb_gray(face):
  img = color.rgb2gray(face)
  return img



# start
detector = mtcnn.MTCNN()
# input video
cap = cv2.VideoCapture(args["video"])
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fpsCap = int(cap.get(cv2.CAP_PROP_FPS))

# arg fps
fpsReduce = args["fps"]

# flg to check apply fps
fpsFlg = True
weight_fps = 1
if fpsReduce <= 0 or fpsReduce >= fpsCap:
  fpsReduce = fpsCap
  fpsFlg = False
else:  
  # weight of fps reduce and fps video
  weight_fps = round(fpsCap / fpsReduce)


# output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args["save"], fourcc, fpsCap, (width,height))


# used to record emotions
result = [0, 0, 0, 0, 0, 0, 0]

# flg to get detect frame
flg = True

# emotions
emotion_dict=('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')

while cap.isOpened():
  # captures frame and returns boolean value and captured image
  ret,test_img=cap.read()
  if not ret:
      continue
  # check frames that are detected
  currentFrame = cap.get(cv2.CAP_PROP_POS_FRAMES)
  # condition FPS
  if fpsFlg:
      if flg:
          firstFrame = currentFrame
          lastFrame = currentFrame + fpsCap

      if currentFrame == lastFrame:
          flg = True
  # detecting ... 
  if (fpsFlg == False) or (currentFrame >= firstFrame and currentFrame < firstFrame + fpsReduce):
    flg = False
    # mtcnn detect face in img
    faces = detector.detect_faces(test_img)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    for face in faces: 
      # only using face image with 95% confidence and detect
      if face['confidence'] > 0.95:
        x, y, w, h = face['box']
        #cropped face
        cropped_face = extract_face(face,test_img)
        # preprocessing img
        image = rgb_gray(cropped_face)
        image = color.gray2rgb(image)
        image = image.reshape((1, 224, 224, 3))

        # predict emotion
        predicted = model.predict(image)
        predicted_class = np.argmax(predicted)
        predicted_percent = predicted[0][predicted_class]
        
        # predict and save result with confidence args
        if (predicted_percent >= float(args['confidence'])):
          # compute result
          if (predicted_class >= 0):
            result[predicted_class] += 1
          
          predicted_label = emotion_dict[predicted_class]
          label = predicted_label + ' ' + str(round(predicted_percent*100)) + '%'
        else:
          label = 'Unknown' 
        
        # write emotion text on frame
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255, 0, 0),thickness=4)                  
        cv2.putText(test_img, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # save video
    resized_img = cv2.resize(test_img, (width, height))
    for i in range(1, weight_fps + 1):
      out.write(resized_img)

  if(currentFrame == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        break

print(*result, sep = ",")
cap.release()
out.release()
cv2.destroyAllWindows