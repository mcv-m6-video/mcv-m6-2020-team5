# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:03:07 2020

@author: Group 5
"""
import cv2
import detectors as dts
from detectors.gt_modifications import obtain_gt
from metrics.mAP import getAvgPrecision
SOURCE = "../datasets/AICity_data/train/S03/c010/vdo.avi"

detectors = {"gt_noise":dts.gt_predict,
             "yolo": dts.yolo_predict,
             "ssd":  dts.ssd_predict,
             "rcnn": dts.rcnn_predict}
def main():
    DETECTOR = "gt_noise"
    cap = cv2.VideoCapture(SOURCE)
    ret, frame = cap.read()
    gt_frames = obtain_gt()
    i = 0
    
    detect_func = detectors[DETECTOR]
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            #predict over the frame
            rects = detect_func(frame)
            
            #Retrack over the frame
            
            #Classify the result
            
            #Obtain GT
            
            #Compute the metrics
            if len(rects[0]) == 4: # No confidence provided
                avg_precision_frame, iou_frame = getAvgPrecision(rects, gt_frames[str(i)])
            if len(rects[0]) == 5: # Provided confidence
                avg_precision_frame, iou_frame = getAvgPrecision(rects, gt_frames[str(i)], True, 1)
            
            #Print Results
            for rect in gt_frames[str(i)]:
                pt1 = (int(rect[0]), int(rect[1]))
                pt2 = (int(rect[2]), int(rect[3]))
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), thickness=2)
            for rect in rects:
                pt1 = (int(rect[0]), int(rect[1]))
                pt2 = (int(rect[2]), int(rect[3]))
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), thickness=2)
            cv2.imshow('Frame',frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            i+=1
        # Break the loop
        else:
            break
    cap.release()
    
    
if __name__ == "__main__":
    main()
    