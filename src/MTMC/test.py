import cv2
import torch
from torch.nn import functional as F
import numpy as np
from gt_from_txt import read_gt,read_det
import os
from tqdm import tqdm
import torchreid
from torchreid.utils import load_pretrained_weights
import os
import pickle
from model import Model
from metrics import calculate_matrices
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
        all_cam_dict[cam] = read_det(annot_path)
        
    return all_cam_dict


def generate_features(all_cam_dict, in_path, sequence_num, camera_list,
                      save_path = "./out/cams", feature_func=None):
    
    if not os.path.exists(save_path):
        os.makedirs(save_paths)
    seq_name = "S{:02d}".format(sequence_num)
    in_path = os.path.join(in_path,seq_name)
    cam_pickles = {}
    for cam in tqdm(camera_list):
        feature_accumulator = {}
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
                    cropped_bbox = frame[int(bb[1]):int(bb[3]),int(bb[0]):int(bb[2])] 
                    feature = feature_func(cropped_bbox)
                    if track_id not in feature_accumulator:
                        feature_accumulator[track_id] = []
                    feature_accumulator[track_id].append(feature)
                    # feature_accumulator.append((track_id,feature))
            i += 1
            pbar.update()
        pbar.close()
        print(f"Saving features for cam {cam}")  
        ppath = os.path.join(save_path, f"{cam}.pkl")
        with open(ppath, "wb+") as f:
            pickle.dump(feature_accumulator, f)
        cam_pickles[cam] = ppath
    return cam_pickles
    
    


if __name__ == "__main__":
    in_path = "../datasets/AIC20_track3_MTMC/test/"
    out_path = "./out/cams"
    sequence = 3
    cameras = [10, 11, 12, 13, 14, 15]
    fc_normalize = False
    load_pickles = False
    
    all_cam_dict = generate_track_for_all_cams(in_path,sequence,cameras)
    
    use_gpu = torch.cuda.is_available()

    dir_to_weights = '../weights/resnet50_triple_10.pth' #Añadir la direccion als weights
    model = Model(dir_to_weights)
    # load_pretrained_weights(model, dir_to_weights)
    ppath = os.path.join(out_path, f"cam_pickles.pkl")
    if not load_pickles:
        cam_pickles = generate_features(all_cam_dict,in_path,sequence,
                                                cameras, save_path=out_path,
                                                feature_func=model)
        with open(ppath, "wb+") as f:
            pickle.dump(cam_pickles, f)
    else:
        # with open(ppath, "rb") as f:
        cam_pickles = pickle.load(open(ppath, "rb"))
        # cam_pickles = {10:os.path.join(out_path, f"10.pkl"),
        #                11:os.path.join(out_path, f"11.pkl"),
        #                12:os.path.join(out_path, f"12.pkl"),
        #                13:os.path.join(out_path, f"13.pkl"),
        #                14:os.path.join(out_path, f"14.pkl"),
        #                15:os.path.join(out_path, f"15.pkl")}
        # with open(ppath, "wb+") as f:
        #     pickle.dump(cam_pickles, f)
    
    
    relation_cams = calculate_matrices(cam_pickles)

    # if fc_normalize:
    #     feature_accumulated_norm = F.normalize(feature_accumulated, p=2, dim=1) #Cambiar para que solo 
    import display
    # display.display_heatmap(relation_cams[10][11])
    # a = "y"
    i = 11
    while  i < 16:
        res = display.display_min(relation_cams[10][i])
        display.print_grid(res, cam_pickles[10], cam_pickles[i])
        # a = input("Continue? y/n")
        cv2.imshow("res", res)
        k = -1
        pressed = False
        while not pressed:
            print("press 'c' to continue")
            k = cv2.waitKey(0)
            print("key:", k)
            if(k == 99):
                i+=1
                pressed = True
    
    
    print()
    