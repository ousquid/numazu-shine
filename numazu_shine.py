# -*- coding=utf-8 -*-
import os
import sys
import numpy as np
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import resnet
import pickle


IMG_HEIGHT = 320
IMG_WIDTH = 240
IMG_IS_COLOR = True

if IMG_IS_COLOR == True:
    IMG_COLOR = 3
else:
    IMG_COLOR = 1
    
def load_misawa():
    path = "/home/ousquid/data/numazu_shine/"
    target_dirs = ["twitter", "instagram"]
    
    x_train, y_train, x_test, y_test = [], [], [], []
    for ans, d in enumerate(target_dirs):
        target_chara = os.path.join(path, d)
        for i, file in enumerate(os.listdir(target_chara)):
            p = os.path.join(target_chara,file)
            img = image.load_img(p, target_size=(IMG_HEIGHT, IMG_WIDTH), grayscale=not(IMG_IS_COLOR))
            imgary = image.img_to_array(img)
            if i%9 == 0:
                x_test.append(imgary)
                y_test.append(ans)
            else:
                x_train.append(imgary)
                y_train.append(ans)

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

with open("numazu.pkl", "rb") as f:
    (x_data, y_data) = pickle.load(f);
    
x_train, y_train, x_test, y_test = [], [], [], []
for i in range(len(x_data)):
    if i % 10 < 3:
        x_test.append(x_data[i])
        y_test.append(y_data[i])
    else:
        x_train.append(x_data[i])
        y_train.append(y_data[i])
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train = np.array(x_train)
y_train = np.array(y_train)
#(x_train, y_train), (x_test, y_test) = load_misawa()

#x_train = x_train.reshape(x_train.shape[0], -1)
#x_test = x_test.reshape(x_test.shape[0], -1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# acc: 18
def simple_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(IMG_HEIGHT*IMG_WIDTH*IMG_COLOR,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def conv_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(11, 11),
                 activation='relu',
                 input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_COLOR)))
    model.add(Conv2D(64, (11, 11), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model
    

#model = simple_model()
#model = conv_model()
model = resnet.ResnetBuilder.build_resnet_18((IMG_COLOR, IMG_HEIGHT, IMG_WIDTH), num_classes)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
              
datagen = image.ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2
)
datagen.fit(x_train)

# HyperParameters
batch_size = 196
steps_per_epoch = 30
epochs = 10
# Callbacks
fp = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
model_check = ModelCheckpoint(fp, monitor='val_loss', save_best_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)



history = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(x_test, y_test),
                     callbacks=[model_check, reduce_lr])
#history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                              steps_per_epoch=steps_per_epoch, 
#                              epochs=epochs,
#                              verbose=1,
#                              validation_data=(x_test, y_test))
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.legend()
plt.savefig("output.png")

model.save("./misawa.h5")

# Convolution2Dにする
# Data Augument（flip, zoom, move）をする
# DeepLearingモデルを使う
