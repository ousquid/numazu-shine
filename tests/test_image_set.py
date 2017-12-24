import unittest
import numpy as np
from image_set import *

class TestImageSet(unittest.TestCase):
    
    def test_initialize(self):
        num_classes, grayscale = None, None
        
        # 正常時
        ok_xdata = np.zeros((10, 128, 256, 1))
        ok_ydata = np.zeros((10, 3))
        
        s = ImageSet(ok_xdata, ok_ydata, num_classes, grayscale)
        self.assertEqual(s.width, 128)
        self.assertEqual(s.height, 256)
        
        # x_data と y_data の要素数が一致しない場合
        lack_ydata = np.zeros((9, 3))
        with self.assertRaises(ValueError):
            ImageSet(ok_xdata, lack_ydata, num_classes, grayscale)
        
        # x_data, y_data で誤った次元数のデータを入れた場合
        ng_xdata = np.zeros((10))
        ng_ydata = np.zeros((10))
        with self.assertRaises(ValueError):
            ImageSet(ng_xdata, ok_ydata, num_classes, grayscale)
        with self.assertRaises(ValueError):
            ImageSet(ok_xdata, ng_ydata, num_classes, grayscale)
        
    
    def test_split(self):
        s = self.get_test_imageset()
        
        # train_rate に合ったデータ数を分割する
        s.split(train_rate=0.9)
        self.assertEqual(len(s.x_train), 9)
        self.assertEqual(len(s.x_test), 1)
        
        s.split(train_rate=0.6)
        self.assertEqual(len(s.x_train), 6)
        self.assertEqual(len(s.x_test), 4)
        
        # seed が異なる場合に違う分割を行う
        s.split(seed="first_seed")
        x_test_first = s.x_test
        s.split(seed="second_seed")
        x_test_second = s.x_test
        self.assertFalse(np.all(x_test_first == x_test_second))
        
        # seed を与えなかった場合に毎回違う分割が行われる
        s.split(train_rate=0.9)
        x_test_tmp = s.x_test
        s.split(train_rate=0.9)
        self.assertFalse(np.all(x_test_tmp == s.x_test))
        
        # train と test にダブリがない
        s.split(train_rate=0.5)
        for data in s.x_train:
            self.assertNotIn(data, s.x_test)
        
        # trainとtestのxとyが対応している
        for i in range(len(s.x_train)):
            self.assertEqual(s.x_train[i][0][0][0], s.y_train[i][0])
        for i in range(len(s.x_test)):
            self.assertEqual(s.x_test[i][0][0][0], s.y_test[i][0])
            
        # train_rate に間違った値を入れたときに例外を返す
        s.split(train_rate=0.0001)
        s.split(train_rate=1)
        with self.assertRaises(ValueError):
            s.split(train_rate=0)
        with self.assertRaises(ValueError):
            s.split(train_rate=1.0001)
        
            
    def get_test_imageset(self):
        x_data = np.zeros((10, 128, 256, 3))
        y_data = np.zeros((10, 3))
        for i in range(10):
            x_data[i].fill(i)
            y_data[i].fill(i)
        num_classes = 3
        grayscale = True
        return ImageSet(x_data, y_data, num_classes, grayscale)
        
        
    def test_get_iter_for_learning_curve(self):
        s = self.get_test_imageset()
        s.split(train_rate=0.9)
        for (i,image_data) in enumerate(s.get_iter_for_learning_curve(9)):
            self.assertEqual(i+1, len(image_data[1]))
            self.assertTrue(np.all(s.x_test == image_data[2]))
            self.assertTrue(np.all(s.y_test == image_data[3]))
        
        with self.assertRaises(ValueError):
            for i in s.get_iter_for_learning_curve(-2):
                break
        with self.assertRaises(ValueError):
            for i in s.get_iter_for_learning_curve(-1):
                break
        with self.assertRaises(ValueError):
            for i in s.get_iter_for_learning_curve(0):
                break
        