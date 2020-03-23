# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:55:34 2018

@author: hamdd
"""

import numpy as np


def linear(img1, img2, **kwargs):
    '''
    Computes linear interpolation between two image objects (numpy arrays)
    INPUT:
        img1, img2: Frames 1 and 2 to be interpolated with
                    - Has to be same size
    OUTPUT:
        imgi: image interpolated (numpy array of same size)
    '''
    
    imgi = np.add(np.divide(img1,2.0), np.divide(img2,2.0))   
    return imgi


