# -*- coding=utf-8 -*-
import os
import sys
import pickle
import numpy as np

import resnet
from image_set import ImageSet, CacheMode
from learning_bot import LearningBot, DrawMethod

def main():
    image_set = ImageSet.load(sys.argv[1], 240, 320)
    image_set.split(train_rate=0.9, seed="numazu_shine")
    
    model = resnet.ResnetBuilder.build_resnet_18(
        (image_set.color, image_set.height, image_set.width), image_set.num_classes)
    model.summary()
    bot = LearningBot(model)

    history = {}
    for image_data in image_set.get_iter_for_learning_curve(5):
        size = len(image_data[0])
        history[size] = bot.learn(*image_data, epochs=30, batch_size=128)

    bot.draw_history_list('test.png', history, ["acc", "val_acc"], method=DrawMethod.best)

if __name__ == "__main__":
    main()