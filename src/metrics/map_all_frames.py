import numpy as np
from random import shuffle
from metrics.mAP import IoU
import copy


def calc_AP(precision, recall):
    precision = np.array(precision)
    recall = np.array(recall)
    ap = 0
    for th in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= th) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= th])
        ap = ap + p / 11.0
    return ap


def frame_AP(n_gt, f_det_bb, frame_gt_bb):
    tp = []
    precision = []
    recall = []
    best_ious = []
    for f_det in f_det_bb[0]:
        ious = []
        correct = False

        if len(frame_gt_bb[0]) == 0:
            break

        for f_gt in frame_gt_bb[0]:
            iou = IoU(f_det, f_gt)
            ious.append(iou)

        iou_bb = np.argmax(ious)
        best_ious.append(ious[iou_bb])
        if ious[iou_bb] > 0.5:
            frame_gt_bb[0].pop(iou_bb)
            correct = True

        tp.append(correct)

        precision.append(tp.count(True) / len(tp))
        recall.append(tp.count(True) / n_gt)

    ap = calc_AP(precision, recall)
    return ap, np.mean(best_ious)


def calculate_ap(det_bb, gt_bb, ini_frame, last_frame, mode):

    lst_gt = [item[0] for item in gt_bb]
    lst_det = [item[0] for item in det_bb]

    AP = 0
    IoU = 0
    for f_val in range(ini_frame, last_frame):
        frame_gt_bb = list([gt_bb[str(i)] for i, num in enumerate(lst_gt) if i == f_val])
        n_gt = len(frame_gt_bb[0])
        frame_det_bb = list([det_bb[str(i)] for i, num in enumerate(lst_det) if i == f_val])

        if mode == 'sort':
            frame_det_bb = sorted(frame_det_bb, key=lambda x: x[-1], reverse=True)
            f_det_bb = [item[:-1] for item in frame_det_bb]
            AP = AP + frame_AP(n_gt, f_det_bb, frame_gt_bb)
        elif mode == 'random':
            
            #Random shuffle
            f_ap = 0
            f_iou = 0
            for i in range(0, 10):
                shuffle(frame_det_bb)
                AP_frame, iou_frame = frame_AP(n_gt, copy.deepcopy(frame_det_bb), copy.deepcopy(frame_gt_bb))
                f_ap = f_ap + AP_frame
                f_iou = f_iou + iou_frame
            AP = AP + f_ap / 10
            IoU = IoU + f_iou / 10
                  
        else:
            #Sorted by area
            frame_det_bb = sorted(frame_det_bb, key=lambda x: (x[5]-x[3])*(x[5]-x[3]), reverse=True)
            AP = AP + frame_AP(n_gt, frame_det_bb, frame_gt_bb)
            
    AP = AP / last_frame
    IoU = IoU / last_frame
    return AP, IoU

