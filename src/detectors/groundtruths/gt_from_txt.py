# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:35:28 2020

@author: hamdd
"""
import numpy as np

SOURCE_GT = "../datasets/AIC20_track3_MTMC/train/"

def obtain_data(sequence, camera):
    global SOURCE_GT
    src = SOURCE_GT + 'S' + f"{sequence:02d}" + '/c' + f"{camera:03d}" + '/gt/gt.txt'
    frame_dict = {}
    with open(src, "r") as opened_file:
        for line in opened_file:
            line_data = line.split(",")
            
            frame_idx = int(line_data[0])-1 #the first frame starts at 0, not 1
            t_id = int(line_data[1])
            pt_x = float(line_data[2])
            pt_y = float(line_data[3])
            width = float(line_data[4])
            height = float(line_data[5])
            label = int(line_data[6])
            
            frame_idx = str(frame_idx)
            if(frame_idx in frame_dict):
                frame_dict[frame_idx].append((pt_x,pt_y,pt_x+width,pt_y+height, t_id, label))
            else:
                frame_dict[frame_idx] = [(pt_x,pt_y,pt_x+width,pt_y+height, t_id, label)]
    return frame_dict
        
def read_gt(seq, camera):
    return obtain_data(seq, camera)

if __name__ == "__main__":
    dict_frame = read_gt(1, 1)
    print(dict_frame)
    