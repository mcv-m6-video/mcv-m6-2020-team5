
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2

import os
import pkg_resources

import detectron2 
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

MEAN_IMAGE = None

class detectron_detector(object):
    def __init__(self,train_frames = 535, weights_path=None, net="retinanet"):
        self._n_of_trainings = 0
        self.thr_n_of_training = train_frames
        self.trained = False
        self.tmp_train_frames = []
        self.weights_path = weights_path
        self.cfg = get_cfg()
        self.predictor = self.__initialize_network(net)
        
    def __initialize_network(self,network):
        retinanet_path = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
        faster_rcnn_path = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
        
        if(detectron2.__version__ == "0.1"):
            if network == "retinanet":
                self.cfg.merge_from_file(pkg_resources.resource_filename("detectron2.model_zoo", os.path.join("configs", retinanet_path)))
                self.cfg.MODEL.WEIGHTS = model_zoo.ModelZooUrls.get(retinanet_path)
                self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
            if network == "faster_rcnn":
                self.cfg.merge_from_file(pkg_resources.resource_filename("detectron2.model_zoo", os.path.join("configs", faster_rcnn_path)))
                self.cfg.MODEL.WEIGHTS = model_zoo.ModelZooUrls.get(faster_rcnn_path)
                self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        else:
            if network == "retinanet":
                self.cfg.merge_from_file(model_zoo.get_config_file(retinanet_path))
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(retinanet_path)
                self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
            if network == "faster_rcnn":
                self.cfg.merge_from_file(model_zoo.get_config_file(faster_rcnn_path))
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(faster_rcnn_path) 
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

            # Create predictor
        return DefaultPredictor(self.cfg)
        
    def train(self,training_frames):
        if(len(training_frames) <= 0):
            raise ValueError("The number of input frames must be bigger than 0")
          
        
        
    def predict(self,frame):
        
        return self.detect(frame)
        
    def detect(self, frame): 
        outputs = self.predictor(frame)
        
        # Visualization in detectron2 framework
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imshow('Image', v.get_image()[:, :, ::-1])
        # cv2.waitKey(0)
        
        # Transformation to bboxes
        bboxes = []
        boxes = outputs['instances'].to("cpu").pred_boxes.tensor.numpy()
        classes = outputs['instances'].to("cpu").pred_classes
        for idx in range(len(classes)):
            if classes[idx] == 0: # Person
                bboxes.append(boxes[idx])    
            if classes[idx] == 2: # Car
                bboxes.append(boxes[idx])              

        return bboxes
    

def obtain_global_var_mean():
    global MEAN_IMAGE
    return np.dstack([MEAN_IMAGE]*3)
        
