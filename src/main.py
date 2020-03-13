# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:03:07 2020

@author: Group 5
"""
import cv2
# import detectors as dts

# from detectors.groundtruths import obtain_gt, obtain_gt_without_static
# from detectors.backgrounds import BGSTModule
from detectors.detectors import obtain_detector
from detectors.groundtruths import obtain_gt

from metrics.mAP import getMetricsClass, IoU
from metrics.graphs import LinePlot, iouFrame
import numpy as np

from display import print_func
from line_arguments import general_parser

from config.utils import obtain_general_config
from detectors.backgrounds.single_gaussian import obtain_global_var_mean
from utils.bbfilters import bbfilters
from metrics.map_all_frames import calculate_ap
from tracking.trackers import obtain_tracker

SOURCE = "../datasets/AICity_data/train/S03/c010/vdo.avi"

def main(new_config):
    gconf = obtain_general_config(gconfig=new_config)

    if(gconf.VISUALIZE):
        w_name = 'display'
        cv2.namedWindow(w_name, cv2.WINDOW_AUTOSIZE)
    # bgsg_module = None
    
    VIDEO_END = gconf.video.start_save+gconf.video.fps*gconf.video.duration
    # cap.set(cv2.CAP_PROP_POS_FRAMES,1450)
    out_cap = None
    cap = cv2.VideoCapture(SOURCE)

    detect_func, bgsg_module = obtain_detector(**gconf.detector)

    gt_frames = obtain_gt(**gconf.gtruth, IoU_func=IoU)

        
    
    avg_precision = []
    iou_history = []
    iou_plot = LinePlot(gconf.plots.iou.name,
                        max_val=gconf.plots.iou.max_val,
                        save_plots=gconf.plots.iou.save)
    # mAP_plot = LinePlot("mAP_frame",max_val=350)
    dt_rects_dict = {}
    
    tracker = obtain_tracker(gconf.tracker.ttype, gconf.tracker)
    i = 0
    while(cap.isOpened() and (not gconf.video.save_video and 
                              (gconf.STOP_AT == -1 or i <= gconf.STOP_AT) or
                              gconf.video.save_video and i <= VIDEO_END)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            #predict over the frame
            print("Frame: ", i)
            # rects = detect_func(frame)
            
            #Retrack over the frame
            
            #Classify the result
            dt_rects = detect_func(frame)
            dt_rects, _ = bbfilters(dt_rects, frame, **gconf.bbox_filter)
            dt_rects = tracker.update(dt_rects)
            dt_rects_dict[str(i)] = list(dt_rects.values())
            #Obtain GT
            
            #Compute the metrics
            avg_precision_frame, iou_frame = getMetricsClass(list(dt_rects.values()), 
                                                             gt_frames[str(i)], 
                                                             nclasses=1)
            if i > gconf.detector.backgrounds.ours.init_at:
                avg_precision.append(avg_precision_frame)
                iou_history.append(iou_frame)
                iou_plot.update(iou_frame)
            #Print Graph

            
            

            # if i > 1000:
            #     break
                # iouFrame(iou_history)
            # iou_plot.update(iou_frame)

            # mAP_plot.update(avg_precision_frame)
            
            #Print Results
            ## prepare data
            gt_rects = gt_frames[str(i)]
            bgseg = None if bgsg_module is None else bgsg_module.get_bgseg()
            orig_bgseg = None if bgsg_module is None else bgsg_module.get_orig_bgseg()

            frame = print_func(frame.copy(), gt_rects, dt_rects, bgseg, orig_bgseg, 
                               gconf.pout, tracker)
            # cv2.imshow('Frame',frame)
            if i > gconf.detector.backgrounds.ours.init_at:
                
                f_out = frame 
                # f_out = obtain_global_var_mean()
                cv2.putText(frame, f"alpha={gconf.detector.backgrounds.ours.alpha}",
                            (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                            2,(255,255,255),6,cv2.LINE_AA)
                if(gconf.video.start_save <= i <= VIDEO_END):
                    if(gconf.video.stack_iou):
                        iou_plot.build_frame(frame)
                    if(gconf.video.save_video):
                        if(out_cap is None):
                            if(gconf.video.stack_iou):
                                fshape = iou_plot.last_img.shape
                            else:
                                fshape = f_out.shape
                            out_cap = cv2.VideoWriter(gconf.video.fname, 
                                                      cv2.VideoWriter_fourcc(*"MJPG"), 
                                                      gconf.video.fps, 
                                                      (fshape[1],fshape[0]))
                    if(gconf.video.stack_iou):
                        f_out = iou_plot.last_img
                    if(gconf.video.save_video):
                        out_cap.write(f_out.astype('uint8'))
                cv2.imshow(w_name, f_out.astype('uint8'))
            
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i+=1
        # Break the loop
        else:
            break

    print("mAP_allframes: {}".format(calculate_ap(dt_rects_dict, gt_frames, 535, 2140, 'random')))
    print("mIoU for all the video: ", np.mean(iou_history))
    print("mAP for all the video: ", np.mean(avg_precision))
    cap.release()
    if(gconf.video.save_video):
        out_cap.release()
    
    
if __name__ == "__main__":
    new_gconfig = []
    parser = general_parser()
    args = parser.parse_args()
    if args.general_config is not None:
        new_gconfig.extend(args.general_config)
    main(new_gconfig)
    
