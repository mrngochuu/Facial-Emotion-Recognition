import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Cut face using haar cascade
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
folder_path = "/Users/mrngochuu/Desktop/Dataset/luan"

for root, dirs, files in os.walk(folder_path, topdown=False):
    for file_name in files:
      if(file_name != '.DS_Store'):
        print('path: ' + str(root) + '/' +str(file_name))
        image = cv2.imread(os.path.join(root,file_name))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30))
        for (x,y,w,h) in faces_detected:
          cv2.rectangle(gray, (x, y), (x+w, y+h), 
                  (0, 255, 0), 2)
          faces = gray[y:y + h, x:x + w]
          cv2.imwrite(os.path.join(root,file_name),faces)
        print('path: ' + str(root) + '/' +str(file_name))


# ImageDataGen
datagen = ImageDataGenerator(
        zoom_range=0.1,
        rescale=1. / 255,
        shear_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.2,2.0],
        fill_mode='nearest')

img = load_img('/Users/mrngochuu/Desktop/disgust/IMG_20210706_172547_541_742758141340829.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/Users/mrngochuu/Desktop/disgust/preview', save_prefix='angry', save_format='jpg'):
    i += 1
    if i > 20:
        break  # 