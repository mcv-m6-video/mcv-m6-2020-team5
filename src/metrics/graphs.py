# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 12:25:19 2020

@author: hamdd
"""

import numpy as np
import matplotlib.pyplot as plt

def iouFrame(iou):
    plt.plot(iou)
    plt.ylim(0,1)
    plt.show()
    