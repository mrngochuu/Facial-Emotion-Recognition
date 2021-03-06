import sys
import os
import numpy as np
import pandas as pd
import tarfile
import shutil
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
# from tensorflow.keras.utils import np_utils
from tensorflow.keras import utils


#unzip file
fname = "fer2013.tar.gz"

if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()
elif fname.endswith("tar"):
    tar = tarfile.open(fname, "r:")
    tar.extractall()
    tar.close()

df = pd.read_csv('fer2013/fer2013.csv')
# print("INFO")
# print(df.info())
# print("Usage")
# print(df["Usage"].value_counts())
# print("HEAD")
# print(df.head())

X_train, train_y, X_test, test_y, X_val, val_y = [], [], [], [], [], []

for index, row in df.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val,'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_val.append(np.array(val,'float32'))
            val_y.append(row['emotion'])
            # X_train.append(np.array(val,'float32'))
            # train_y.append(row['emotion'])
        elif 'PrivateTest' in row['Usage']:
            X_test.append(np.array(val,'float32'))
            test_y.append(row['emotion'])
    except:
        print("error occurred at index: ", {index}," and row: ", {row})

# print("X_train sample data :",X_train[0:2])
# print("train_y sample data :",train_y[0:2])
# print("X_test sample data :",X_test[0:2]) 
# print("test_y sample data :",test_y[0:2])

X_train=np.array(X_train,'float32')
train_y=np.array(train_y,'float32')
X_test=np.array(X_test,'float32')
test_y=np.array(test_y,'float32')
X_val=np.array(X_val,'float32')
val_y=np.array(val_y,'float32')

# Normalizing data between 0 and 1.

# Mean normalizing
# X_train-=np.mean(X_train,axis=0)
# X_train/=np.std(X_train,axis=0)

# X_test-=np.mean(X_test,axis=0)
# X_test/=np.std(X_test,axis=0)

# X_val-=np.mean(X_val,axis=0)
# X_val/=np.std(X_val,axis=0)

# Min-Max normalizing
X_train-=np.min(X_train, axis=0)
X_train/=(np.max(X_train, axis=0) - np.min(X_train, axis=0))

X_test-=np.min(X_test, axis=0)
X_test/=(np.max(X_test, axis=0) - np.min(X_test, axis=0))

X_val-=np.min(X_val, axis=0)
X_val/=(np.max(X_val, axis=0) - np.min(X_val, axis=0))

num_features = 64
num_labels = 7
batch_size = 64
epochs = 50
width, height = 48, 48

X_train = X_train.reshape(X_train.shape[0], width, height, 1)
X_test = X_test.reshape(X_test.shape[0], width, height, 1)
X_val = X_val.reshape(X_val.shape[0], width, height, 1)

train_y = utils.to_categorical(train_y, num_classes=num_labels)
test_y = utils.to_categorical(test_y, num_classes=num_labels)
val_y = utils.to_categorical(val_y, num_classes=num_labels)

#designing in cnn
model = Sequential()

# MODEL - 01
# #1st layers
# model.add(Conv2D(num_features, kernel_size = (3,3), activation = 'relu', input_shape=(X_train.shape[1:])))
# model.add(Conv2D(num_features, kernel_size = (3,3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.5))

# #2nd convolutional layer
# model.add(Conv2D(num_features, (3,3), activation='relu'))
# model.add(Conv2D(num_features, (3,3), activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))
# model.add(Dropout(0.5))

# #3rd convolutional layer
# model.add(Conv2D(2*num_features, (3,3), activation='relu'))
# model.add(Conv2D(2*num_features, (3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size = (2,2), strides=(2,2)))

# model.add(Flatten())

# model.add(Dense(2*2*2*2*num_features, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(2*2*2*2*num_features, activation='relu'))
# model.add(Dropout(0.2))

# model.add(Dense(num_labels, activation='softmax'))

# MODEL 02
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# model.summary()
# model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
# model.make_model()
# model.load_weights()
# callback = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss', min_delta=0, patience=5,
#     mode='auto', restore_best_weights=True
# )
model.compile(loss=categorical_crossentropy,optimizer=Adam(),metrics=['accuracy'])

history = model.fit(X_train, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, val_y), shuffle=True)
hist_df = pd.DataFrame(history.history)

test_loss, test_acc = model.evaluate(X_test, test_y, verbose=2)
# save accuracy
acc_txt_file = "acc.txt"
with open(acc_txt_file, mode='w') as f:
    f.write(str(test_acc))
print('\nTest accuracy: ', test_acc)

#Create a plot of accuracy and loss over time
history_dict = history.history
history_dict.keys()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('trainingLoss.png')

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('traningAcc.png')

#saving model
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")

# save history file
hist_json_file = 'history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

#remove file
dir_path = 'fer2013'

try:
    shutil.rmtree(dir_path)
except OSError as e:
    print("Error: %s : %s" % (dir_path, e.strerror))

