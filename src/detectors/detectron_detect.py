
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

MEAN_IMAGE = None

class detectron_detector(object):
    def __init__(self,train_frames = 535, weights_path=None):
        self._n_of_trainings = 0
        self.thr_n_of_training = train_frames
        self.trained = False
        self.tmp_train_frames = []
        self.weights_path = weights_path
        self.predictor = self.__initialize_network()
        
    def __initialize_network(self):
        cfg = get_cfg()
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = "../models/retina_net_R50/model_final.pkl"

        # Create predictor
        return DefaultPredictor(cfg)
        
    def train(self,training_frames):
        if(len(training_frames) <= 0):
            raise ValueError("The number of input frames must be bigger than 0")
          
        
        # We put the images into a stack so numpy operations are easier. Shape (w,h,n_frame)
        # training_frames = np.stack(training_frames,axis=2)
        
        print("we are here bois")
        
        
    def predict(self,frame):
        # if(self._n_of_trainings < self.thr_n_of_training):
        #     self.tmp_train_frames.append(frame)
        #     self._n_of_trainings+=1
        #     return []
        # else:
        #     if(not self.trained):
        #         self.train(self.tmp_train_frames)
        #         self.trained = True
        #         self.tmp_train_frames.clear()
        #         del self.tmp_train_frames
        return self.detect(frame)
        
    def detect(self, frame):
        if(self.mean_image is None or self.variance_image is None):
            raise ValueError("The background model is not correctly initializated \
                                the train function must be called to do so")     
        boxes = self.predictor(frame)
        
        
        return None
    

def obtain_global_var_mean():
    global MEAN_IMAGE
    return np.dstack([MEAN_IMAGE]*3)
        
