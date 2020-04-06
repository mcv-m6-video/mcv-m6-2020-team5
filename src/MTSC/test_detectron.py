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
from metrics.mot import mot_metrics
import numpy as np

from display import print_func
from line_arguments import general_parser

from config.utils import obtain_general_config
from detectors.backgrounds.single_gaussian import obtain_global_var_mean
from utils.bbfilters import bbfilters, bbfilters_remove_static
from metrics.map_all_frames import calculate_ap
from tracking.trackers import obtain_tracker
from tqdm import tqdm 
from collections import OrderedDict
import opflows.visualization as opt_view
from MTSC.gt_from_txt import read_gt, read_det

import pickle

import glob
import os
import copy
import MTSC.detectron_detect_multicameras as dt

# IMPORTANT: Change SEQUENCE and CAMERA according to the ones you want to test
SOURCE = "../datasets/AIC20_track3_MTMC/test/"
SEQUENCE = 3
# CAMERA = 15
NUMBER_FRAMES = "../datasets/AIC20_track3_MTMC/cam_framenum/S" + f"{SEQUENCE:02d}" + '.txt'

def test_detectron(new_config, CAMERA, REGISTERED):
    SRC = SOURCE + 'S' + f"{SEQUENCE:02d}" + '/c' + f"{CAMERA:03d}"
    
    gconf = obtain_general_config(gconfig=new_config)

    if(gconf.display.frames):
        w_name = 'display'
        cv2.namedWindow(w_name, cv2.WINDOW_AUTOSIZE)
    # bgsg_module = None
    
    VIDEO_END = gconf.video.start_save+gconf.video.fps*gconf.video.duration
    # cap.set(cv2.CAP_PROP_POS_FRAMES,1450)

    gt_frames = read_gt(SRC)
    # dt_frames = read_det(SRC)
    # gt_frames = obtain_gt(**gconf.gtruth, IoU_func=IoU)
    keys = []
    
    for key in gt_frames.keys():
        keys.append(int(key))
    
    # detect_func, bgsg_module = obtain_detector(**gconf.detector, gt_frames=gt_frames)
    dclass = dt.detectron_detector_multicameras(net=gconf.detector.detectron.net, reg=REGISTERED)
    detect_func = dclass.predict
    bgsg_module = None

    tracking_metrics = mot_metrics()
    avg_precision = []
    iou_history = []

    if(not gconf.display.iou_plot):
        import matplotlib.pyplot as plt
        plt.ion()
    create_iou = gconf.display.iou_plot or gconf.plots.iou.save or \
        gconf.video.save_video and gconf.video.stack_iou
    if(create_iou):
        iou_plot = LinePlot(gconf.plots.iou.name,
                            max_val=gconf.plots.iou.max_val,
                            save_plots=gconf.plots.iou.save)
    
    # Dictionaries to compute the AP over all the frames together
    dt_rects_dict = {}
    gt_rects_dict = {}
    
    # Get the values to use for training and validation
    pkl_file_train = open('../datasets/detectron2/dataset_train_S01_S04.pkl', 'rb')
    pkl_file_val = open('../datasets/detectron2/dataset_val_S03.pkl', 'rb')

    dataset_train = pickle.load(pkl_file_train, fix_imports=True, encoding='ASCII', errors='strict')
    dataset_val = pickle.load(pkl_file_val, fix_imports=True, encoding='ASCII', errors='strict')
    imgs_in_dataset_val = [d['file_name'].split('/')[-1] for d in dataset_val]
    
    tracker = obtain_tracker(gconf.tracker.ttype, gconf.tracker)
    
    if(gconf.detector.dtype == 'detectron'):
        INIT_DISPLAY = 0
    else:
        INIT_DISPLAY = gconf.detector.backgrounds.ours.init_at
    with open (NUMBER_FRAMES, 'rt') as number_frames:
        for line in number_frames:
            if line.split(' ')[0] == "c" + f"{CAMERA:03d}":   
                total_frames = int(line.split(' ')[1])      
    pbar = tqdm(total=total_frames)
    
    src = os.path.join(SRC, "vdo.avi")
    out_cap = None
    cap = cv2.VideoCapture(src)
    i = 0
    nval_img = 0 
    while(cap.isOpened() and (not gconf.video.save_video and 
                            (gconf.STOP_AT == -1 or i <= gconf.STOP_AT) or
                            gconf.video.save_video and i <= VIDEO_END)):

        
        # Capture frame-by-frame
        if(i > 0):
            old_frame = non_modified_frame
        ret, frame = cap.read()
        if(ret):
            non_modified_frame = frame.copy()
        if(i == 0):
            old_frame = non_modified_frame
        
        pbar.update()
        if ret == True:
            if( i > gconf.START_PROCESSING_AT ):
                
                if(gconf.tracker.ttype == "optical_flow_track"):
                    gray_img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(old_frame,gray_img1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Predict over the frame
                dt_rects = detect_func(frame)
                # dt_rects = dt_frames[str(i)]

                
                # Remove static cars
                if i > 5:
                    dt_rects, dt_rects_last = bbfilters_remove_static(dt_rects, dt_rects_last, i)
                else:
                    dt_rects, dt_rects_last = bbfilters_remove_static(dt_rects, None, i)

                # Filter bboxes
                dt_rects, _ = bbfilters(dt_rects, frame, **gconf.bbox_filter)
                orig_dt_rects = dt_rects.copy()
                
                # Retrack over the frame
                if(gconf.tracker.ttype == "optical_flow_track"):
                    dt_rects = tracker.update(dt_rects,flow)
                else:
                    dt_rects = tracker.update(dt_rects)
                                
                # Obtain GT
                if i not in keys:
                    gt_rects = []
                else:
                    gt_rects = gt_frames[str(i)]
                
                # Include detections and gt to compute the metrics
                compute_metric = False

                if gt_rects or dt_rects:
                    compute_metric = True
                if i > 5 and compute_metric:
                    dt_rects_dict[str(nval_img)] = list(dt_rects.values())
                    gt_rects_dict[str(nval_img)] = gt_rects
                    nval_img += 1
                    
                # Include detections for tracking
                if i > 5 and compute_metric:
                    dt_track = OrderedDict()
                    for dt_id, dtrect in dt_rects.items():
                        dt_track.update({dt_id: tracker.object_paths[dt_id]})
                        
                    tracking_metrics.update(dt_track,gt_rects)
                                       
               
                bgseg = None if bgsg_module is None else bgsg_module.get_bgseg()
                orig_bgseg = None if bgsg_module is None else bgsg_module.get_orig_bgseg()

                if(gconf.pout.activate and (gconf.display.frames or gconf.video.save_video)):
                    frame = print_func(frame.copy(), gt_rects, dt_rects, 
                                    orig_dt_rects, bgseg, orig_bgseg, 
                                    gconf.pout, tracker)
                # cv2.imshow('Frame',frame)
                if i > INIT_DISPLAY:
                    
                    f_out = frame 
                    # f_out = obtain_global_var_mean()
                    # cv2.putText(frame, f"alpha={gconf.detector.backgrounds.ours.alpha}",
                    #             (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                    #             2,(255,255,255),6,cv2.LINE_AA)
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
                    if(gconf.display.frames):
                        cv2.imshow(w_name, f_out.astype('uint8'))
                        
                # if(gconf.tracker.ttype == "optical_flow_track"):
                #     flow_rgb = opt_view.colorflow_white(flow)
                #     cv2.imshow("flow",flow_rgb)
                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            i+=1
        # Break the loop
        else:
            break
        
    mAP, mIoU = calculate_ap(dt_rects_dict, gt_rects_dict, 0, len(gt_rects_dict), 'random')
    print("mAP: ", mAP)
    print("mIoU: ", mIoU)
    
    idf1, idp, idr = tracking_metrics.get_metrics()
    print("idf1: ", idf1)
    
    camera_output = 'camera_' + str(CAMERA) + '.txt'
    
    with open (camera_output, 'w') as out_file:
        out_file.write("mAP: " + str(mAP) + "\n")
        out_file.write("mIoU: " + str(mIoU) + "\n")
        out_file.write("idf1: " + str(idf1) + "\n")
        
    cap.release()
    if(gconf.video.save_video):
        out_cap.release()
    
    
if __name__ == "__main__":
    parser = general_parser()
    args = parser.parse_args()
    new_gconfig = []
    configs_jj = []

    if args.general_config is not None:
        new_gconfig.extend(args.general_config)

    main(new_gconfig)
    
