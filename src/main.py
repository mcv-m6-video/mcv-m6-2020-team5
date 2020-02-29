# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:03:07 2020

@author: Group 5
"""
import cv2
from detectors.gt_modifications import predict as gt_predict
from detectors.gt_modifications import obtain_gt
from metrics.mAP import getPRCurve
SOURCE = "../datasets/AICity_data/train/S03/c010/vdo.avi"
def main():
    
    cap = cv2.VideoCapture(SOURCE)
    ret, frame = cap.read()
    gt_frames = obtain_gt()
    i = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            #predict over the frame
            rects = gt_predict(frame)
            
            #Retrack over the frame
            
            #Classify the result
            
            #Obtain GT
            
            #Compute the metrics
            getPRCurve(rects, gt_frames[str(i)])
            #Print Results
            for rect in rects:
                pt1 = (int(rect[0]), int(rect[1]))
                pt2 = (int(rect[2]), int(rect[3]))
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), thickness=5)
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
    