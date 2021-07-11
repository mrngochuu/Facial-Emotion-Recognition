import cv2
import os
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# folder_path = "/Users/mrngochuu/Desktop/Dataset/emotion"

# for root, dirs, files in os.walk(folder_path, topdown=False):

#     for file_name in files:
#       if(file_name != '.DS_Store'):
#         image = cv2.imread(os.path.join(root,file_name))
#         print('root: ' + str(root))
#         print('image: ' + str(image))
#         print('filename: ' + str(file_name))
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         cv2.imwrite(os.path.join(root,file_name),gray)


train_data_dir = "/Users/mrngochuu/Desktop/Dataset/emotion/train"
datagen = ImageDataGenerator(rescale=1. / 255,
                             shear_range=0.3,
                             zoom_range=0.2,
                             horizontal_flip=True)
train_generator = datagen.flow_from_directory(train_data_dir,
                                              target_size=(224,224),
                                              batch_size=4,
                                              color_mode="rgb",
                                              class_mode='categorical')

test_data_dir = "/Users/mrngochuu/Desktop/Dataset/emotion/test"
datagen = ImageDataGenerator(rescale=1. / 255,
                             shear_range=0.3,
                             zoom_range=0.2,
                             horizontal_flip=True)
test_generator = datagen.flow_from_directory(test_data_dir,
                                             target_size=(224,224),
                                             batch_size=4,
                                             color_mode="rgb",
                                             class_mode='categorical')

vggmodel = VGGFace(model='vgg16', include_top=True)
vggmodel.summary()

for layers in (vggmodel.layers)[:19]:
    print(layers)
    layers.trainable = False

X= vggmodel.layers[-2].output
predictions = Dense(7, activation="softmax")(X)
model_final = Model(input = vggmodel.input, output = predictions)

model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model_final.summary()

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=7, verbose=1, mode='auto')
stats = model_final.fit_generator(generator= train_generator,epochs= 1, validation_data= test_generator, callbacks=[checkpoint,early])
model_final.save("emotion_prediction_vggface_5.h5")


plt.plot(stats.history["acc"])
plt.plot(stats.history['val_acc'])
plt.plot(stats.history['loss'])
plt.plot(stats.history['val_loss'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["Accuracy","Validation Accuracy","Loss","Validation Loss"])
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 12
plt.rcParams["figure.figsize"] = fig_size
plt.savefig('traningAcc.png')
plt.show()