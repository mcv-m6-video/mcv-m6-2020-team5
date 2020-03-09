# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:40:15 2020

@author: hamdd
"""

import cv2
import numpy as np
import os

# from .subsense.Python.Subsense import Subsense, Lobster
from .single_gaussian import gausian_back_remov
from .color_gaussian import color_gausian_back_remov

cv2mag, cv2med, cv3min = cv2.__version__.split(".")

class BGSTModule(object):
    def __init__(self, downsample=1, scene_mask_path="", bs_type = "MOG2",history = 1000, varThreshold=30, detectShadows=True,rho=0, alpha=1, trigger=0, init_at = 535, color_space = 'GRAY',*args, **kwargs):
        
        # super(BGSGThread, self).__init__(img_buffer, *args, **kwargs)
        
        self.d = downsample
        self.scene_mask_path = scene_mask_path
        
        self.bs_type = bs_type
        self.rho = rho
        self.alpha = alpha
        
        self.init_at = init_at
        
        self.history = history
        self.varThreshold = varThreshold
        self.detectShadows = detectShadows
        
        self.scene_mask = None
        self.fgbg = None
        
        self.color_space = color_space
        
        
        self.kern3 = None
        self.kern5 = None
        self.kernV = None

        self.initialization()
        
        self.orig_bgseg = None
        self.last_bgseg = None
    def get_orig_bgseg(self):
        return self.orig_bgseg
    def get_bgseg(self):
        return self.last_bgseg
    def initialization(self):
        # scene_mask = cv2.imread(self.scene_mask_path,0)
        
        d = self.d
        self.scene_mask = None
        if(not self.scene_mask_path == ""):
            scene_mask = cv2.imread(self.scene_mask_path,0)
            if(scene_mask is None):
                msg = "Scene Mask could not be loaded!"
                msg+= "\n    Path: {}".format(self.scene_mask_path)
                msg+= "\n    Does OS find the file?:{}".format(os.path.isfile(self.scene_mask_path))
                raise(IOError(msg))
            self.scene_mask = scene_mask[0::d,0::d]
        if(self.bs_type == "MOG"):
            self.fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=100)
        elif(self.bs_type == "MOG2"):
            self.fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30, detectShadows=True)
            self.fgbg.setShadowValue(0)
        elif(self.bs_type == "CNT"):
            self.fgbg = cv2.bgsegm.createBackgroundSubtractorCNT()
        elif(self.bs_type == "GMG"):
            self.fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
        elif(self.bs_type == "LSBP"):
            self.fgbg = cv2.bgsegm.createBackgroundSubtractorLSBP()
        elif(self.bs_type == "GSOC"):
            self.fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
        elif(self.bs_type == "gauss_black_rem"):
            self.fgbg = gausian_back_remov(self.rho, self.alpha, self.init_at,self.color_space)        
        elif(self.bs_type == "color_gauss_black_rem"):
            self.fgbg = color_gausian_back_remov(self.rho, self.alpha, self.init_at, self.color_space)
        # elif(self.bs_type == "Subsense"):
        #     self.fgbg = Subsense()
        # elif(self.bs_type == "Lobster"):
        #     self.fgbg = Lobster()
            
            
        self.kern1 = np.ones((5,5),np.uint8)
        self.kern2 = np.ones((7,7),np.uint8)
        self.kern3 = np.ones((15,9),np.uint8)
    
    def get_contours(self, frame):
        results = []
        d = self.d
        d = 1
        dframe = frame[0::d,0::d,:]
        dframe = cv2.GaussianBlur(dframe, (7, 7), 0)
        if(self.bs_type ==  "CNT"):
            fmask = self.fgbg.apply(dframe, learningRate=0.99)
        else:
            fmask = self.fgbg.apply(dframe)
        if(fmask is None):
            fmask = np.zeros(frame.shape[:2], dtype=np.uint8)
        res = cv2.bitwise_and(fmask,fmask,mask = self.scene_mask)
        res[res<127] = 0
        res[res>=127] = 255
        
        cv2.imshow("mask direct", res)
        
        self.orig_bgseg = res
        
        morph = res
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, self.kern1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN,  self.kern2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, self.kern3)
        # cv2.imshow("blur frame", dframe)
        # cv2.imshow("bgsg test", morph)
        binary = cv2.bitwise_and(morph,morph)
        self.last_bgseg = binary
        if(int(cv2mag) > 3):
            contours,__ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            __,contours,__ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
           (x,y,w,h) = cv2.boundingRect(cnt)
           x, y, w, h = x*d, y*d, w*d, h*d
           # x, y = x+w/2, y+h/2
           # if(w*h > 2000):
               # results.append(("blob", 0.9, (x, y, w, h)))
           results.append((x, y, x+w, y+h))
        return results