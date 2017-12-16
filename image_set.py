# -*- coding=utf-8 -*-
import os
import random
import pickle
import hashlib
from keras.preprocessing import image
from enum import Enum
import numpy as np
import keras
import math


CacheMode = Enum("CacheMode", "use rewrite ignore")

class ImageSet(object):
    def __init__(self, x_data, y_data, num_classes, grayscale, width, height):
        self.x_data = x_data
        self.y_data = y_data
        self.num_classes = num_classes
        self.color = 1 if grayscale else 3
        self.width = width
        self.height = height

    @staticmethod
    def load(path, width, height, *, grayscale=True, cache=CacheMode.use):
        def get_md5(path, height, width, grayscale):
            key = "".join(map(str, [path, height, width, grayscale]))
            filename = hashlib.md5(key.encode('utf-8')).hexdigest() + ".pkl"
            return filename

        # num classes
        target_dirs = os.listdir(path)
        num_classes = len(target_dirs)

        # open cache
        file_name = get_md5(path, height, width, grayscale)
        pklfile = os.path.join("/tmp/", file_name)
        if cache==CacheMode.use and os.path.exists(pklfile):
            with open(pklfile, "rb") as f:
                (tmp_x, tmp_y) = pickle.load(f);
            return ImageSet(tmp_x, tmp_y, num_classes, grayscale, width, height)

        # load images
        tmp_x, tmp_y = [], []
        for ans, d in enumerate(target_dirs):
            target_chara = os.path.join(path, d)
            for i, file in enumerate(os.listdir(target_chara)):
                p = os.path.join(target_chara,file)
                img = image.load_img(p, target_size=(height, width), grayscale=grayscale)
                imgary = image.img_to_array(img)
                
                tmp_x.append(imgary)
                tmp_y.append(ans)

        tmp_x, tmp_y = np.array(tmp_x), np.array(tmp_y)
        tmp_x.astype('float32')
        tmp_x /= 255
        tmp_y = keras.utils.to_categorical(tmp_y, num_classes)
        tmp_inst = ImageSet(tmp_x, tmp_y, num_classes, grayscale, width, height)

        # save cache
        if cache != CacheMode.ignore:
            with open(pklfile, "wb") as f:
                pickle.dump((tmp_inst.x_data, tmp_inst.y_data), f, protocol=pickle.HIGHEST_PROTOCOL)
        return tmp_inst

    def split(self, train_rate=0.9, seed=None):
        # init rand mod.
        random.seed(seed)
        
        # calc data size.
        datsz = len(self.x_data)
        partsz = math.floor(datsz * train_rate)
        
        # calc index.
        rand_idx = list(range(datsz))
        random.shuffle(rand_idx)
        train_idx = rand_idx[:partsz]
        test_idx = rand_idx[partsz:]
        
        # pack data.
        self.x_train = np.array([self.x_data[i] for i in train_idx])
        self.y_train = np.array([self.y_data[i] for i in train_idx])
        self.x_test = np.array([self.x_data[i] for i in test_idx])
        self.y_test = np.array([self.y_data[i] for i in test_idx])

    def get_iter_for_learning_curve(self, times):
        image_data_list = []
        for num in np.linspace(0, len(self.x_train), times, dtype=int):
            if num == 0:
                continue
            
            idxs = random.sample(range(len(self.x_train)), num)
            x_train = [self.x_train[i] for i in idxs]
            y_train = [self.y_train[i] for i in idxs]
            yield(x_train, y_train, self.x_test, self.y_test)

    def __str__(self):
        return "ImageSet"