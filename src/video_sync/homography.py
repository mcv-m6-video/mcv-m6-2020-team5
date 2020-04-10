#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:39:19 2020

@author: dazmer
"""
import cv2
import numpy as np
from dataset_video import cameraSync

from test_sync import get_starting_video_offsets

import glob

def string_to_mat(string):
    rows = string.split(";")
    rnps = [np.fromstring(r, dtype=np.float, sep=" ") for r in rows]
    rnps = np.array(rnps) if len(rnps)>1 else rnps[0]
    
    return rnps

def read_calibration_matrix(calib_file):
    # print("Calib file:", calib_file)
    with open(calib_file,"r") as file:
         data = file.read()
         data = data.replace("\n",":")
         # data = data.replace('Homography matrix:', '&')
         # data = data.replace('Intrinsic parameter matrix:', '&')
         # data = data.replace('Intrinsic parameter matrix:', '&')
         # print("DATA:", data)
         splits = data.split(":")
         homat = string_to_mat(splits[1])
         # iparm = string_to_mat(splits[3])
         # distr = string_to_mat(splits[5])
         # reprj = string_to_mat(splits[7])
         # homat[:2,2] = 0
    return homat
         
def test(seq):
    seq_path = f"../datasets/AIC20_track3_MTMC/train/{seq}"
    seq_path_full = glob.glob(seq_path+"/*")
    video_path_list = [v+"/vdo.avi" for v in seq_path_full]
    calib_path_list = [c+"/calibration.txt" for c in seq_path_full]
    
    v_offsets_text_path = f"../datasets/AIC20_track3_MTMC/cam_timestamp/{seq}.txt"
    v_offsets = get_starting_video_offsets(v_offsets_text_path)
    video_offset_list = np.array(v_offsets)
    
    calib_list = [read_calibration_matrix(cp) for cp in calib_path_list]
    
    minval = np.max(video_offset_list[1:])
    video_offset_list = minval - video_offset_list
    video_offset_list[:] = 80000
    
    video_list = []
    for path,offset in zip(video_path_list,video_offset_list):
        video_list.append(cameraSync(path,offset_ms=offset))

    lframes = [c.read()[1] for c in video_list]
    while(all([f is not None for f in lframes])):
        for i,frame in enumerate(lframes):
            hmat = calib_list[i]
            # frame = cv2.resize(frame,(480,270))
            ftransf = cv2.warpPerspective(frame, np.linalg.inv(hmat), (100,100))
            # ftransf = cv2.resize(ftransf,(10,10))
            cv2.imshow(str(i),frame)
            cv2.imshow(f"h{i}", ftransf)
        cv2.waitKey(100) 

        lframes = [c.read()[1] for c in video_list]


    
if __name__ == "__main__":
    seq = "S01"
    # rcm = read_calibration_matrix(f"../datasets/AIC20_track3_MTMC/cam_timestamp/{seq}.txt")
    test(seq)
    pass