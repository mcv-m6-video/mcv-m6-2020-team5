
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2


class color_gausian_back_remov(object):
    def __init__(self, rho, alpha, thr_n_trainings=125):
        self.mean_image = None
        self.variance_image = None
        self.rho = rho
        self.alpha = alpha
        self._n_of_trainings = 0
        self.thr_n_of_training = thr_n_trainings
        self.trained = False
        self.tmp_train_frames = []
        
    def train(self,training_frames):
        if(len(training_frames) <= 0):
            raise ValueError("The number of input frames must be bigger than 0")
        
        # We put the images into a stack so numpy operations are easier. Shape (w,h,n_frame)
        training_frames = np.stack(training_frames,axis=3)
        
        self.mean_image = np.mean(training_frames,axis=3)
        self.variance_image = np.std(training_frames,axis=3)
        
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
        
        # we check both sides of the gaussian to see if it's inside
        positive_variance = frame <(self.mean_image+self.alpha*(self.variance_image+2)) 
        negative_variance = frame >(self.mean_image-self.alpha*(self.variance_image+2)) 
        
        # if the values are inbetween the thresholds it's background 
        is_background = (positive_variance*negative_variance)
        is_foreground = np.logical_not(is_background)
        
        # we now decide if it's background or foreground
        # is_background,is_foreground = self.__voting_system(is_background,is_foreground)
        is_background,is_foreground = self.__unanimity_for_background(is_background,is_foreground)
        
        self.__update_mean_image(frame,is_background,is_foreground)
        self.__update_variance_image(frame,is_background,is_foreground)
        
        return (is_foreground[:,:,0]*255).astype("uint8")
    
    def __voting_system(self,is_background,is_foreground):
        dim = is_background.shape[2]
        background_votes = np.sum(is_background,axis=2)
        foreground_votes = np.sum(is_foreground,axis=2)
        
        ret_is_background = background_votes > foreground_votes
        ret_is_foreground = np.logical_not(ret_is_background)
        
        return np.dstack( ((ret_is_background),)*dim ),np.dstack( ((ret_is_foreground),)*dim )
    
    def __unanimity_for_background(self,is_background,is_foreground):
        dim = is_background.shape[2]
        
        ret_is_background = np.prod(is_background,axis=2).astype("bool")
        ret_is_foreground = np.logical_not(ret_is_background)
        
        return np.dstack( ((ret_is_background),)*dim ),np.dstack( ((ret_is_foreground),)*dim )
        
    # def __convert_array_to_gray(self,frame_array):
    #     for frame_idx in range(len(frame_array)):
    #         frame_array[frame_idx] = cv2.cvtColor(frame_array[frame_idx],cv2.COLOR_BGR2GRAY)
    #         # temp = cv2.cvtColor(frame_array[frame_idx],cv2.COLOR_BGR2YUV)
    #         # frame_array[frame_idx] = temp[:,:,0]
    #     return frame_array
    
    def __update_mean_image(self,frame,background_mask,foreground_mask):
        # we update only the pixels that are background
        new_mean_image = background_mask*( (1-self.rho)*self.mean_image+self.rho*frame )
        new_mean_image = new_mean_image + foreground_mask*self.mean_image
        self.mean_image = new_mean_image
    
    def __update_variance_image(self,frame,background_mask,foreground_mask):
        new_variance = np.std(np.stack((frame,self.mean_image),axis=3),axis=3)+2
        # we update only the pixels that are background
        new_variance_image = background_mask*( (1-self.rho)*self.variance_image+self.rho*new_variance )
        # new_variance_image = background_mask*( (1-self.rho)*self.variance_image+self.rho*frame)       
        new_variance_image = new_variance_image + foreground_mask*self.variance_image
        self.variance_image = new_variance_image

        
        
        
