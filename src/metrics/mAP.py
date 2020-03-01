# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 19:25:19 2020

@author: hamdd
"""
import numpy as np
from copy import deepcopy
import tensorflow as tf
from tensorflow.compat.v1.metrics import average_precision_at_k


# get Average Precision and Average IoU for each frame
def getAvgPrecision(pred_bboxes, gt_bboxes, confidence=False, N=10, IoU_threshold=0.5):
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
    prediction = []
    iou = []
    for x in range(N):
        if confidence:
            sorted(pred_bboxes, key=lambda pred_bboxes: pred_bboxes[4])
        else:
            np.random.shuffle(rrank)

        tgt_bboxes = deepcopy(gt_bboxes)
        corrects = []
        for idx in rrank:
            if len(tgt_bboxes) is not 0:
                pdbbox = pred_bboxes[idx][0:4]
                res = [IoU(pdbbox, tgt_bboxes[i]) for i in range(len(tgt_bboxes))]
                idx_max = np.where(res==np.max(res))[0][0]
                conf_max = res[idx_max]
                iou.append(conf_max)
                if conf_max > IoU_threshold:
                    corrects.append(1)
                    tgt_bboxes.remove(tgt_bboxes[idx_max])
                else:
                    corrects.append(0)
            else:
                iou.append(0)
                corrects.append(0)
                
        while len(corrects) < len(gt_bboxes):
            corrects.append(0)
        prediction.append(corrects)
    
    avg_precision = mAP(prediction, len(gt_bboxes))
    avg_iou = sum(iou) / len(iou)

    return avg_precision, avg_iou

#Compute mAP given a prediction vector and the length of ground truth boxes 
def mAP(prediction, len_gt_bboxes):
    """
        Calculate mAP from a given prediction
        (Not completed yet)
    """
    labels = tf.ones(len_gt_bboxes, dtype=tf.dtypes.int64)
    ap_sum = tf.zeros(1, dtype=tf.dtypes.int64)
    for pred in range(len(prediction)):
        predictions = tf.convert_to_tensor(prediction[pred], dtype=tf.dtypes.int64, dtype_hint=None)
        ap = average_precision_at_k(labels, predictions, len_gt_bboxes)
        ap_sum = tf.math.add(ap_sum, ap)
    map = ap_sum / len(prediction)
    # print(predictions)
    return map


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