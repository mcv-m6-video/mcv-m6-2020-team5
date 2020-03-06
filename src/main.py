# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:03:07 2020

@author: Group 5
"""
import cv2
import detectors as dts
from detectors.gt_modifications import obtain_gt
from detectors.backgrounds import BGSTModule
from metrics.mAP import getMetricsClass
from metrics.graphs import LinePlot, iouFrame
import numpy as np

SOURCE = "../datasets/AICity_data/train/S03/c010/vdo.avi"

detectors = {"gt_noise":dts.gt_predict,
             "yolo": dts.yolo_predict,
             "ssd":  dts.ssd_predict,
             "rcnn": dts.rcnn_predict,
             "MOG2": None,
             "CNT": None}
def main():

    SAVE_PLOT_IOU = True
    SAVE_EVERY_X_FRAME = 1
    STOP_AT = 500
    DETECTOR = "Subsense"
    
    det_backgrounds = ["MOG", "MOG2", "CNT", "GMG", "LSBP", "GSOC", "Subsense", "Lobster"]
    if(DETECTOR in det_backgrounds):
        bgsg_module = BGSTModule(bs_type = DETECTOR)
        f = bgsg_module.get_contours
        for d in det_backgrounds:
            detectors[d] = f
        
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
            
            #Obtain GT
            
            #Compute the metrics
            avg_precision_frame, iou_frame = getMetricsClass(rects, gt_frames[str(i)], nclasses=1)
            avg_precision.append(avg_precision_frame)
            iou_history.append(iou_frame)
            #Print Graph


            iou_plot.update(iou_frame)

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
                iou_plot.save_plot(cv2.resize(frame, None, fx=0.3, fy=0.3))
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
    
