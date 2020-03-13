
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2

MEAN_IMAGE = None

class gausian_back_remov(object):
    def __init__(self, rho, alpha, thr_n_trainings=535, channel = 'GRAY'):
        self.mean_image = None
        self.variance_image = None
        self.rho = rho
        self.alpha = alpha
        self._n_of_trainings = 0
        self.thr_n_of_training = thr_n_trainings
        self.trained = False
        self.tmp_train_frames = []
        self.channel = channel
    def train(self,training_frames):
        if(len(training_frames) <= 0):
            raise ValueError("The number of input frames must be bigger than 0")
          
        training_frames = self.__convert_array_to_gray(training_frames)
        # We put the images into a stack so numpy operations are easier. Shape (w,h,n_frame)
        training_frames = np.stack(training_frames,axis=2)
        
        self.mean_image = np.mean(training_frames,axis=2)
        self.variance_image = np.std(training_frames,axis=2)
    # def train(self,frame):
          
    #     # training_frames = self.__convert_array_to_gray(training_frames)
    #     # We put the images into a stack so numpy operations are easier. Shape (w,h,n_frame)
    #     # training_frames = np.stack(training_frames,axis=2)
    #     gframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #     if(self.mean_image is None):
    #         self.mean_image = np.zeros_like(gframe)
    #     self.mean_image += np.divide(gframe, self.thr_n_of_training)
        
    #     # self.mean_image = np.mean(training_frames,axis=2)
    #     self.variance_image = np.std(training_frames,axis=2)
        
    def apply(self,frame):
        if(self._n_of_trainings < self.thr_n_of_training):
            self.tmp_train_frames.append(frame)
            self._n_of_trainings+=1
        else:
            if(not self.trained):
                self.train(self.tmp_train_frames)
                self.trained = True
                self.tmp_train_frames.clear()
                del self.tmp_train_frames
            return self.substract(frame)
    def substract(self, frame):
        if(self.mean_image is None or self.variance_image is None):
            raise ValueError("The background model is not correctly initializated \
                                the train function must be called to do so")
        
        if self.channel == 'GRAY':
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        elif self.channel == 'HUE':
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,0]
        elif self.channel == 'L':
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2Lab)[:,:,0]
        elif self.channel == 'Y':
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)[:,:,0]
        elif self.channel == 'SATURATION':
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,1]
        else:
            raise ValueError("The color channel selected is not valid. Choose a valid color channel, please.")            
            
        
        # we check both sides of the gaussian to see if it's inside
        positive_variance = gray_frame <(self.mean_image+self.alpha*(self.variance_image+2)) 
        negative_variance = gray_frame >(self.mean_image-self.alpha*(self.variance_image+2)) 
        
        # if the values are inbetween the thresholds it's background 
        is_background = (positive_variance*negative_variance)
        is_foreground = np.logical_not(is_background)
        self.__update_mean_image(gray_frame,is_background,is_foreground)
        self.__update_variance_image(gray_frame,is_background,is_foreground)
        cv2.imwrite("plots/mean_image.png", self.mean_image)
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
        global MEAN_IMAGE
        MEAN_IMAGE = self.mean_image
    
    def __update_variance_image(self,frame,background_mask,foreground_mask):
        new_variance = np.std(np.dstack((frame,self.mean_image)),axis=2)+2
        # we update only the pixels that are background
        new_variance_image = background_mask*( (1-self.rho)*self.variance_image+self.rho*new_variance )
        # new_variance_image = background_mask*( (1-self.rho)*self.variance_image+self.rho*frame)       
        new_variance_image = new_variance_image + foreground_mask*self.variance_image
        self.variance_image = new_variance_image

def obtain_global_var_mean():
    global MEAN_IMAGE
    return np.dstack([MEAN_IMAGE]*3)
        
