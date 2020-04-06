import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2

import os
import pkg_resources

import detectron2 
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import cv2
from detectors.groundtruths.gt_modifications import obtain_gt
from tqdm import tqdm
from PIL import Image
import random
import pickle

import xmltodict
from MTSC.gt_from_txt import read_gt

MEAN_IMAGE = None

import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

def string_sort(string_list):
    string_list.sort(key=natural_keys)

class detectron_detector_multicameras(object):
    def __init__(self,train_frames = 3000, weights_path=None, net=None, 
                 training = 'True', objects=["bike","car"], reg=None):
        self._n_of_trainings = 0
        self.thr_n_of_training = train_frames
        self.training = training
        self.registered = reg
        self.tmp_train_frames = []
        self.weights_path = weights_path
        self.cfg = get_cfg()
        self.predictor = self.__initialize_network(net)
        self.dobjects = objects
        
    def __initialize_network(self,network):
        if self.training:
            self.train(self.thr_n_of_training, network) 
        retinanet_path = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
        faster_rcnn_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        
        if(detectron2.__version__ == "0.1"):
            if network == "retinanet":
                self.cfg.merge_from_file(pkg_resources.resource_filename("detectron2.model_zoo", os.path.join("configs", retinanet_path)))
                if not self.training:
                    self.cfg.MODEL.WEIGHTS = model_zoo.ModelZooUrls.get(retinanet_path)
                else:
                    self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')
                self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
                
            if network == "faster_rcnn":
                self.cfg.merge_from_file(pkg_resources.resource_filename("detectron2.model_zoo", os.path.join("configs", faster_rcnn_path)))
                if not self.training:
                    self.cfg.MODEL.WEIGHTS = model_zoo.ModelZooUrls.get(faster_rcnn_path)
                else:
                    self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')
                self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        else:
            if network == "retinanet":
                self.cfg.merge_from_file(model_zoo.get_config_file(retinanet_path))
                if not self.training:
                    self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(retinanet_path)
                else:
                    self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')
                self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5  # set threshold for this model
            if network == "faster_rcnn":
                self.cfg.merge_from_file(model_zoo.get_config_file(faster_rcnn_path))
                if not self.training:
                    self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(faster_rcnn_path) 
                else:
                    self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')
                self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

            # Create predictor
        return DefaultPredictor(self.cfg)
      
        
    def generate_datasets(self, Ntraining):
        dataset_train, dataset_val = get_dicts(Ntraining)
        if not self.registered:
            for d in ['train', 'val']:
                DatasetCatalog.register(d + '_set', lambda d=d: dataset_train if d == 'train' else dataset_val)
                MetadataCatalog.get(d + '_set').set(thing_classes=['Person', 'None', 'Car'])

    def train(self, training_frames, network):

        if(training_frames <= 0):
            raise ValueError("The number of input frames must be bigger than 0")
        
        self.cfg.OUTPUT_DIR = (f'../datasets/detectron2/{network}')
    
        self.generate_datasets(training_frames)
            
        retinanet_path = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
        faster_rcnn_path = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        if(detectron2.__version__ == "0.1"):
            if network == 'faster_rcnn':
                self.cfg.merge_from_file(pkg_resources.resource_filename("detectron2.model_zoo", os.path.join("configs", faster_rcnn_path)))
                self.cfg.MODEL.WEIGHTS = model_zoo.ModelZooUrls.get(faster_rcnn_path)
            if network == 'retinanet':
                self.cfg.merge_from_file(pkg_resources.resource_filename("detectron2.model_zoo", os.path.join("configs", retinanet_path)))
                self.cfg.MODEL.WEIGHTS = model_zoo.ModelZooUrls.get(retinanet_path)
        else:
            if network == 'faster_rcnn':
                self.cfg.merge_from_file(model_zoo.get_config_file(faster_rcnn_path))
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(faster_rcnn_path) 
            if network == 'retinanet':
                self.cfg.merge_from_file(model_zoo.get_config_file(retinanet_path))
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(retinanet_path) 

        self.cfg.DATASETS.TRAIN = ('train_set',)
        self.cfg.DATASETS.TEST = ('val_set',)
        self.cfg.DATALOADER.NUM_WORKERS = 1
        self.cfg.SOLVER.IMS_PER_BATCH = 1
        self.cfg.SOLVER.BASE_LR = 0.001
        self.cfg.SOLVER.MAX_ITER = 2000
        self.cfg.SOLVER.STEPS = (1000, 2000)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.DEVICE='cuda' 
        
        if not os.path.isfile(os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')):
        
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
            
            trainer = DefaultTrainer(self.cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
            
            # evaluator = COCOEvaluator("val_set", self.cfg, False, output_dir=self.cfg.OUTPUT_DIR)
            # trainer.test(self.cfg, trainer.model, evaluators=[evaluator])
            # trainer.test(self.cfg, trainer.model)  
            # self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')
                
        
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
        boxes = outputs['instances'].to("cpu").pred_boxes.tensor.numpy(); 
        classes = outputs['instances'].to("cpu").pred_classes
        if self.training:
            label2id =  {"car":0, "bike":1}
        else:
            label2id =  {"car":2, "bike":0}
        
        id2label = {v: k for k, v in label2id.items()}
        # print(classes)
        for idx in range(len(classes)):
            iD = int(classes[idx])
            if iD in id2label and id2label[iD] in self.dobjects:
                bboxes.append(boxes[idx])   
        return bboxes
    

def obtain_global_var_mean():
    global MEAN_IMAGE
    return np.dstack([MEAN_IMAGE]*3)

def get_dicts(N_frames):
    img_dir_train="../datasets/AIC20_track3_MTMC/train/"
    img_dir_test="../datasets/AIC20_track3_MTMC/test/"
    annot_dir="../datasets/"
    output_dir_train="frames_train/"
    output_dir_test="frames_test/"
    
    dataset_train = []
    dataset_val = []
    
    train_dir = annot_dir + f'detectron2/dataset_train_S01_S04.pkl'
    val_dir   = annot_dir + f'detectron2/dataset_val_S03.pkl'
                
    scene_dirs = []
    scene_fds = os.listdir(img_dir_train)
    for scene_fd in scene_fds:
        scene_dirs.append(os.path.join(img_dir_train, scene_fd))
    
    for scene_dir in scene_dirs:
        camera_dirs = []
        fds = os.listdir(scene_dir)
        for fd in fds:
            if fd.startswith('c0'):
                camera_dirs.append(os.path.join(scene_dir, fd))
        for camera_dir in camera_dirs:
            print(camera_dir)
            gt_frames = read_gt(camera_dir)   
            output_dir_train_camera = os.path.join(camera_dir,output_dir_train)
            print(output_dir_train_camera)   
            # if not os.path.exists(output_dir_train_camera):  
            #     os.mkdir(output_dir_train_camera)
                    
            #     filename = os.path.join(camera_dir, "vdo.avi")
            #     vidcap = cv2.VideoCapture(filename)
            #     success,image = vidcap.read()
            #     count = 0
                
            #     while success:
            #         cv2.imwrite(output_dir_train_camera + "frame_%d.jpg" % count, image)     # save frame as JPEG file      
            #         success,image = vidcap.read()
            #         count += 1
                
            
            keys = []
                
            for key in gt_frames.keys():
                keys.append(int(key))
            
            
            if not os.path.isfile(train_dir):
                   
                image_list = os.listdir(output_dir_train_camera)
                image_list.sort(key=natural_keys)
                
                dataset_dicts = []
                
                for idx, img_name in tqdm(enumerate(image_list),desc='Getting dicts'):
                    
                    if idx not in keys:
                        pass
                    else:
                        boxes = gt_frames[str(idx)]

                        record = {}
                        filename = os.path.join(output_dir_train_camera, img_name)
                        width, height = Image.open(filename).size

                        record["file_name"] = filename
                        record["image_id"] = idx
                        record["height"] = height
                        record["width"] = width

                        objs = []
                        
                        for coord in range(len(boxes)):
                            bbox = [float(xy) for xy in boxes[coord][0:4]]
                            label = boxes[coord][5]
                            cat_id = 0
                            obj = {
                                "bbox": bbox,
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "category_id": cat_id,
                                "iscrowd": 0
                                }
                            objs.append(obj)

                        record["annotations"] = objs
                        dataset_dicts.append(record)
                        
                dataset_train += dataset_dicts
                # print(len(dataset_dicts))
                # print(len(dataset_train))

            else:
                pkl_file_train = open(train_dir, 'rb')
                dataset_train = pickle.load(pkl_file_train, fix_imports=True, encoding='ASCII', errors='strict')
                    
    with open(train_dir, 'wb') as handle:
        pickle.dump(dataset_train, handle)               
                
                 
            
    scene_dirs_test = []
    scene_fds_test = os.listdir(img_dir_test)
    for scene_fd_test in scene_fds_test:
        scene_dirs_test.append(os.path.join(img_dir_test, scene_fd_test))
    
    for scene_dir_test in scene_dirs_test:
        camera_dirs_test = []
        fds_test = os.listdir(scene_dir_test)
        for fd in fds_test:
            if fd.startswith('c0'):
                camera_dirs_test.append(os.path.join(scene_dir_test, fd))      
        for camera_dir in camera_dirs_test:
            print(camera_dir)
            gt_frames = read_gt(camera_dir)   
            output_dir_test_camera = os.path.join(camera_dir,output_dir_test)   
            # if not os.path.exists(output_dir_test_camera): 
            #     os.mkdir(output_dir_test_camera)
 
            #     filename = os.path.join(camera_dir, "vdo.avi")
            #     vidcap = cv2.VideoCapture(filename)
            #     success,image = vidcap.read()
            #     count = 0
                
            #     while success:
            #         cv2.imwrite(output_dir_test_camera + "frame_%d.jpg" % count, image)     # save frame as JPEG file      
            #         success,image = vidcap.read()
            #         count += 1
                        
            keys = []
                
            for key in gt_frames.keys():
                keys.append(int(key))

            if not os.path.isfile(val_dir):
                                
                image_list = os.listdir(output_dir_test_camera)
                image_list.sort(key=natural_keys)
                
                dataset_dicts = []
                
                for idx, img_name in tqdm(enumerate(image_list),desc='Getting dicts'):
                    if idx not in keys:
                        pass
                    else:
                        boxes = gt_frames[str(idx)]

                        record = {}
                        filename = os.path.join(output_dir_test_camera, img_name)
                        width, height = Image.open(filename).size

                        record["file_name"] = filename
                        record["image_id"] = idx
                        record["height"] = height
                        record["width"] = width

                        objs = []
                        
                        for coord in range(len(boxes)):
                            bbox = [float(xy) for xy in boxes[coord][0:4]]
                            label = boxes[coord][5]
                            cat_id = 0
                            obj = {
                                "bbox": bbox,
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "category_id": cat_id,
                                "iscrowd": 0
                                }
                            objs.append(obj)

                        record["annotations"] = objs
                        dataset_dicts.append(record)

                dataset_val += dataset_dicts  
                # print(len(dataset_dicts))
                # print(len(dataset_val))

            else:
                pkl_file_val = open(val_dir, 'rb')
                dataset_val = pickle.load(pkl_file_val, fix_imports=True, encoding='ASCII', errors='strict')    
                
    with open(val_dir, 'wb') as handle:
        pickle.dump(dataset_val, handle)
  
    print(len(dataset_train))
    print(len(dataset_val))
    return dataset_train, dataset_val
   