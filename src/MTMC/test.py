import cv2
import numpy as np
from gt_from_txt import read_gt
import os
from tqdm import tqdm

# classifies the dictionary by track instead of by frames
def regenerate_tracks(camera_dict):
    track_dict = {}
    
    for obj_frame,id_frame in zip(camera_dict.values(),camera_dict.keys()):
        id_frame = int(id_frame)
        for obj in obj_frame:
            bb = obj_frame[0][0:4]
            bb_id = obj_frame[0][4]
            
            if(bb_id in track_dict):
                track_dict[bb_id][id_frame] = bb  
            else:
                dict_temp = {}
                track_dict[bb_id] = dict_temp
                track_dict[bb_id][id_frame] = bb 
             
    return track_dict

def generate_track_for_all_cams(in_path, sequence_num, camera_list):
    
    all_cam_dict = {}
    seq_name = "S{:02d}".format(sequence_num)
    in_path = os.path.join(in_path,seq_name)
    for cam in camera_list:
        cam_name = "c{:03d}".format(cam)
        annot_path = os.path.join(in_path,cam_name)
    
        # temp_dict = read_gt(annot_path)
        # all_cam_dict[cam] = regenerate_tracks(temp_dict)
        all_cam_dict[cam] = read_gt(annot_path)
        
    return all_cam_dict

def dummy_feature_predict(frame_in,bb):
    
    cropped_frame = frame_in[int(bb[1]):int(bb[3]),int(bb[0]):int(bb[2])] 
    
    # cv2.imshow("test",cropped_frame)
    # cv2.waitKey(1)
    
    #do some predict features in hereeee

    #return track_id and feature vector
    return []


def generate_features(all_cam_dict, in_path, sequence_num, camera_list):
    
    feature_accumulator = []
    
    seq_name = "S{:02d}".format(sequence_num)
    in_path = os.path.join(in_path,seq_name)
    for cam in tqdm(camera_list):
        cam_name = "c{:03d}".format(cam)
        cam_path = os.path.join(in_path,cam_name)
        video_path = os.path.join(cam_path,"vdo.avi")
        current_cam_dict = all_cam_dict[cam]
        
        current_cap = cv2.VideoCapture(video_path)
        total_frames = int(current_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pbar = tqdm(total=total_frames)
        i = 0
        ret = current_cap.isOpened()
        while(ret):
            ret, frame = current_cap.read()
            
            if(i in current_cam_dict):
                bb_list = current_cam_dict[i]
                for bb_info in bb_list:
                    bb = bb_info[0:4]
                    track_id = bb_info[4]
                    feature = dummy_feature_predict(frame,bb)
                    feature_accumulator.append((track_id,i,feature))
            i += 1
            pbar.update()
        pbar.close()
            
    return feature_accumulator
    
            

if __name__ == "__main__":
    in_path = "../datasets/AIC20_track3_MTMC/test/"
    sequence = 3
    cameras = [10, 11, 12, 13, 14, 15]
    
    all_cam_dict = generate_track_for_all_cams(in_path,sequence,cameras)
    generate_features(all_cam_dict,in_path,sequence,cameras)
    print()
    