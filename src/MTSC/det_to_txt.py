# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:35:28 2020

@author: hamdd
"""
import numpy as np

# def obtain_data(sequence, camera):
def get_det(track_id, det, frame_id, path, label):
    frame_dict = {}
    print(frame_id)
    print(det)
    with open(path, "r+") as opened_file:
        for line in opened_file:
            pass
        opened_file.write(str(frame_id) + ',' + str(track_id) + ',' + f'{det[0]:.2f}' + ',' +  f'{det[1]:.2f}' + ',' +  f'{det[2]-det[0]:.2f}' + ',' + f'{det[3]-det[1]:.2f}'   + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + '\n')
    

def write_det(dt_id, dt_rects, frame, path, network):
    src = path + '/det/det_' + network + '.txt'
    return get_det(dt_id, dt_rects, frame, src, False)

# if __name__ == "__main__":
#     dict_frame = read_gt(1, 1)
#     print(dict_frame)
    