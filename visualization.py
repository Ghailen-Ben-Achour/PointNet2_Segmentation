# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:12:13 2021

@author: msi
"""

import numpy as np
import laspy
import pptk
import os
import glob


path_to_predictions="./test"




POINT_SIZE = 0.001



    
if __name__ == '__main__':



    for f in (glob.glob(os.path.join(path_to_predictions,'*.npy'))):
        #print(f,s)		

        points = np.load(f)
        print(points.shape)

        data1 = points[:,:3]
        label1 = points[:,3]

        rgb_codes = [[200, 90, 0],
            [255, 0, 0],
            [255, 0, 255],
            [0, 220, 0],
            [0, 200, 255]]

        color = np.zeros((label1.shape[0], 3))
        for i in range(label1.shape[0]):
            color[i,:] = [code/255 for code in rgb_codes[int(label1[i])]]

        v = pptk.viewer(data1[:,:3],color)
        v.set(point_size=POINT_SIZE,show_axis=False,show_info=False)

        v.wait()

        v.close()
