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
import resnet
import pickle


IMG_HEIGHT = 320
IMG_WIDTH = 240
IMG_IS_COLOR = True

if IMG_IS_COLOR == True:
    IMG_COLOR = 3
else:
    IMG_COLOR = 1
    
def load_numazu():
    path = "/home/ousquid/data/numazu_shine/"
    target_dirs = ["twitter", "instagram"]
    
    x_data, y_data = [], []
    for ans, d in enumerate(target_dirs):
        target_chara = os.path.join(path, d)
        for i, file in enumerate(os.listdir(target_chara)):
            p = os.path.join(target_chara,file)
            img = image.load_img(p, target_size=(IMG_HEIGHT, IMG_WIDTH), grayscale=not(IMG_IS_COLOR))
            imgary = image.img_to_array(img)
            x_data.append(imgary)
            y_data.append(ans)

    return np.array(x_data), np.array(y_data)


data = load_numazu()
with open("numazu.pkl", "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
