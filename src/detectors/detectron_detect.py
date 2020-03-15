
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
import cv2
from detectors.groundtruths.gt_modifications import obtain_gt

import xmltodict

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
    
    
        
    def generate_datasets(self, Ntraining):
        
        DatasetCatalog.register('training_set', get_dicts('train', Ntraining))
        MetadataCatalog.get('training_set').set(thing_classes='Car')
        
    def train(self,training_frames):
        if(len(training_frames) <= 0):
            raise ValueError("The number of input frames must be bigger than 0")
        
        self.generate_datasets(len(training_frames), method = 'random')
        
        #Crear el dataloader per al dataset d'entrenament.
        self.cfg.DATASETS.TRAIN = ('training_set',)#Modificar
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.001
        self.cfg.SOLVER.MAX_ITER = 15000
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(kitti_mots_dataset.thing_classes)#Modificar

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()    
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')
        #Fer el train a aquí.Tot i això cal crear els dicts com fem al M5  
        
        return self.cfg
        
        
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

def get_dicts(lists = 'train', N=0.25, method = 'random'):
    img_dir="../datasets/AICity_data/train/S03/c010" 
    annot_dir="../datasets/"
    
    filename = os.path.join(img_dir, "vdo.avi")
    annotname = os.path.join(img_dir, "ai_challenge_s03_c010-full_annotation.xml")

    dataset_dicts = []
    dataset_train = []
    dataset_val = []
    
    thing_classes = ["Car"]
    
    if not os.path.exists(dirName):
        vidcap = cv2.VideoCapture(filename)
        success,image = vidcap.read()
        count = 0
    
        dirName = "frames"
        os.mkdir(dirName)
        
        while success:
            cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1
    
    for idx, img_name in tqdm(enumerate(os.listdir("../frames/")),desc='Getting dicts'):
        record = {}
        filename = os.path.join("../frames", img_name)
        height, width = Image.open(filename).size

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        #Refer les annotations
        with open(annotname, "rb") as fd:
            gdict = xmltodict.parse(fd)

        objs = []
        
        frame_list = obtain_gt(include_parked = True)
                
        for coord in frame_list:
            
            bbox = [float(xy) for xy in coord]
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": thing_classes[0],
                "iscrowd": 0
                }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    dataset_train = dataset_dicts
    
        #Separar entre training i test (comprovar que estigui bé que és un copy paste molt a saco)
    if method == 'random':
        val_samples = random.choices(np.arange(0,len(self),1),k=int(len(self)*split))
        val_img = [str(img).zfill(6)+'.png' for img in val_samples]
        dataset_train = [dic for dic in dataset_dicts if not dic['file_name'].split('/')[-1] in val_img]
        dataset_val = [dic for dic in dataset_dicts if dic['file_name'].split('/')[-1] in val_img]
        
    return dataset_train, dataset_val
    