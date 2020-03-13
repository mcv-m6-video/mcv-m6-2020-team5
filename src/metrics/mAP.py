# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 19:25:19 2020

@author: hamdd
"""
import numpy as np
from copy import deepcopy


# multi class metrics
def getMetricsClass(pred_bboxes, gt_bboxes, nclasses):
    """
    Get average precision and intersection over union across different classes 
    
    """
    aps = []
    iou = []
    for cls in range(nclasses):
        if bool(pred_bboxes):
            if len(pred_bboxes[0]) == 4: 
                avg_precision_class, iou_class = getMetrics(pred_bboxes, gt_bboxes)
            if len(pred_bboxes[0]) == 5:
                avg_precision_class, iou_class = getMetrics(pred_bboxes, gt_bboxes, confidence = True)
        else:
            avg_precision_class = 0
            iou_class = 0

        aps.append(avg_precision_class)
        iou.append(iou_class)
        
    return np.mean(aps), np.mean(iou)
        
        
# get Average Precision and Average IoU for each frame for only one class
def getMetrics(pred_bboxes, gt_bboxes, confidence=False, N=10, IoU_threshold=0.5):
    """
    Input: 
        pred_bboxes: predicted bounding boxes
        gt_bboxes: ground truth bounding boxes
        confidence: True if confidence is provided in the predicted boxes
        N: number of random shuffle if there is no confidence provided
        IoU_threshold: value to consider the IoU a correct bounding box
    
    Output:
        avg_precision: mean Average Precision of the given predicted boxes
        avg_iou: average IoU of the given predicted boxes
    
    """
    rrank = np.arange(len(pred_bboxes))
    aps = []
    iou = []
    if confidence:
        N = 1
    for x in range(N):
        # Sort predicted boxes by confidence or randomly
        if confidence:
            sorted(pred_bboxes, key=lambda pred_bboxes: pred_bboxes[4])
        else:
            np.random.shuffle(rrank)

        tgt_bboxes = deepcopy(gt_bboxes)
        tp = np.zeros(len(rrank))
        fp = np.zeros(len(rrank))
        for idx in rrank:
            if len(tgt_bboxes) is not 0:
                pdbbox = pred_bboxes[idx][0:4]
                IoU_list = [IoU(pdbbox, tgt_bboxes[i]) for i in range(len(tgt_bboxes))]
                idx_max = np.where(IoU_list==np.max(IoU_list))[0][0]
                max_IoU = IoU_list[idx_max]
                iou.append(max_IoU)
                # Get tp and fp comparing with Iou threshold
                if max_IoU > IoU_threshold:
                    tp[idx] = 1
                    tgt_bboxes.remove(tgt_bboxes[idx_max])
                else:
                    fp[idx] = 1
            else:
                iou.append(0)
                fp[idx] = 1
        ap = average_precision(tp,fp,len(gt_bboxes))
        aps.append(ap)
        
    avg_precision = np.mean(aps)
    avg_iou = np.mean(iou)
    
    return avg_precision, avg_iou


#Compute Average Precision given a prediction vector and the length of ground truth boxes 
def average_precision(tp,fp,npos):
    """
    Calculate average precision from the given tp, fp and number of measurements
    Code modification from Detectron2: pascal_voc_evaluation.py 
    (https://github.com/facebookresearch/detectron2/blob/master/detectron2/evaluation/pascal_voc_evaluation.py)
    Input:
        tp: true positives bounding boxes
        fp: false positives bounding boxes
        npos: number of measurements  
    Output:
        ap: average precision
    """
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        
    # compute VOC AP using 11 point metric
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.0

    return ap


def IoU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou