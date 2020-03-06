# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:03:07 2020

@author: Group 5
"""
import cv2
import detectors as dts
from detectors.single_gaussian import gausian_back_remov
from detectors.gt_modifications import obtain_gt
from metrics.mAP import getMetricsClass
from metrics.graphs import LinePlot, iouFrame
import numpy as np

SOURCE = "../datasets/AICity_data/train/S03/c010/vdo.avi"

detectors = {"gt_noise":dts.gt_predict,
             "yolo": dts.yolo_predict,
             "ssd":  dts.ssd_predict,
             "rcnn": dts.rcnn_predict}
def main():

    SAVE_PLOT_IOU = True
    SAVE_EVERY_X_FRAME = 1
    STOP_AT = 500
    NUM_OF_TRAINING_FRAMES = 100
    
    background_removal = gausian_back_remov(0.01,2.5)
    training_frames = []


    DETECTOR = "yolo"

    cap = cv2.VideoCapture(SOURCE)
    # cap.set(cv2.CAP_PROP_POS_FRAMES,1450)
    ret, frame = cap.read()
    gt_frames = obtain_gt()
    i = 0
    avg_precision = []
    iou_history = []
    iou_plot = LinePlot("IoU_frame",max_val=300, save_plots=SAVE_PLOT_IOU)
    # mAP_plot = LinePlot("mAP_frame",max_val=350)
    detect_func = detectors[DETECTOR]
    while(cap.isOpened() and (STOP_AT == -1 or i <= STOP_AT)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            #predict over the frame
            print("Frame: ", i)
            rects = detect_func(frame)
            
            #Retrack over the frame
            
            #Classify the result
            
            if(i == NUM_OF_TRAINING_FRAMES):
                background_removal.train(training_frames)
            else:
                if(i < NUM_OF_TRAINING_FRAMES):
                    training_frames.append(np.copy(frame))
                else:
                    back_remov_frame = background_removal.apply(frame)
                    cv2.imshow("removed",back_remov_frame)
            
            #Obtain GT
            
            #Compute the metrics
            # avg_precision_frame, iou_frame = getMetricsClass(rects, gt_frames[str(i)], nclasses=1)
            # avg_precision.append(avg_precision_frame)
            # iou_history.append(iou_frame)
            #Print Graph


            # iou_plot.update(iou_frame)

            # if i == 500:
                # iouFrame(iou_history)
            # iou_plot.update(iou_frame)

            # mAP_plot.update(avg_precision_frame)
            
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
            if(SAVE_PLOT_IOU and (i%SAVE_EVERY_X_FRAME)==0):
                # iou_plot.save_plot(cv2.resize(frame, None, fx=0.3, fy=0.3))
                pass

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            i+=1
        # Break the loop
        else:
            break
    print("mIoU for all the video: ", np.mean(iou_history))
    print("mAP for all the video: ", np.mean(avg_precision))
    cap.release()
    
    
if __name__ == "__main__":
    main()
    
