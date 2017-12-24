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
        with self.assertRaises(NumazuShineException):
            ImageSet(ok_xdata, lack_ydata, num_classes, grayscale)
        
        # x_data, y_data で誤った次元数のデータを入れた場合
        ng_xdata = np.zeros((10))
        ng_ydata = np.zeros((10))
        with self.assertRaises(NumazuShineException):
            ImageSet(ng_xdata, ok_ydata, num_classes, grayscale)
            ImageSet(ok_xdata, ng_ydata, num_classes, grayscale)
