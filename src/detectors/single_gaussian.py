
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2






class gausian_back_remov(object):
    def __init__(self, rho, thrs):
        self.mean_image = None
        self.variance_image = None
        self.rho = rho
        self.thrs = thrs
    
    def train(self,training_frames):
        if(len(training_frames) <= 0):
            raise ValueError("The number of input frames must be bigger than 0")
          
        training_frames = self.__convert_array_to_gray(training_frames)
        # We put the images into a stack so numpy operations are easier. Shape (w,h,n_frame)
        training_frames = np.stack(training_frames,axis=2)
        
        self.mean_image = np.mean(training_frames,axis=2)
        self.variance_image = np.var(training_frames,axis=2)
        
    def apply(self,frame):
        if(self.mean_image is None or self.variance_image is None):
            raise ValueError("The background model is not correctly initializated \
                                the train function must be called to do so")
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        # we check both sides of the gaussian to see if it's inside
        positive_variance = gray_frame <(self.mean_image+self.thrs*self.variance_image) 
        negative_variance = gray_frame >(self.mean_image-self.thrs*self.variance_image) 
        
        # if the values are inbetween the thresholds it's background 
        is_background = (positive_variance*negative_variance)
        is_foreground = np.logical_not(is_background)
        
        self.__update_mean_image(gray_frame,is_background,is_foreground)
        self.__update_variance_image(gray_frame,is_background,is_foreground)
        
        return (is_foreground*255).astype("uint8")
    
    
    def __convert_array_to_gray(self,frame_array):
        for frame_idx in range(len(frame_array)):
            frame_array[frame_idx] = cv2.cvtColor(frame_array[frame_idx],cv2.COLOR_BGR2GRAY)
        return frame_array
    
    def __update_mean_image(self,frame,background_mask,foreground_mask):
        # we update only the pixels that are background
        new_mean_image = background_mask*( (1-self.rho)*self.mean_image+self.rho*frame )
        new_mean_image = new_mean_image + foreground_mask*self.mean_image
        self.mean_image = new_mean_image
    
    def __update_variance_image(self,frame,background_mask,foreground_mask):
        # we update only the pixels that are background
        new_variance_image = background_mask*( (1-self.rho)*self.variance_image+self.rho*frame )
        new_variance_image = new_variance_image + foreground_mask*self.variance_image
        self.variance_image = new_variance_image

        
        
        
