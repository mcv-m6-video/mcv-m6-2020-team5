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
from collections import OrderedDict
import glob

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
                      save_path = "./out/cams", feature_func=None,
                      return_images = False, max_permitted_size = 150*150*3):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    seq_name = "S{:02d}".format(sequence_num)
    in_path = os.path.join(in_path,seq_name)
    cam_pickles = {}
    img_pickles = {}
    print("CAM LIST:", camera_list)
    for cam in tqdm(camera_list):
        feature_accumulator = OrderedDict()
        if(return_images):
            img_accumulator = OrderedDict()
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
                        if(return_images):
                            img_accumulator[track_id] = []
                    feature_accumulator[track_id].append(feature)
                    if(return_images):
                        if(cropped_bbox.size > max_permitted_size):
                            rf = np.sqrt(max_permitted_size/cropped_bbox.size)
                            cropped_bbox = cv2.resize(cropped_bbox,None, fx=rf, fy=rf)
                        img_accumulator[track_id].append(cropped_bbox)
                    # feature_accumulator.append((track_id,feature))
            i += 1
            pbar.update()
        pbar.close()
        
        print(f"Saving features for cam {cam}")  
        ppath = os.path.join(save_path, f"{cam}.pkl")
        with open(ppath, "wb+") as f:
            pickle.dump(feature_accumulator, f)
        cam_pickles[cam] = ppath
            
        if(return_images):
            print(f"Saving images for cam {cam}")  
            ppath2 = os.path.join(save_path, f"{cam}_imgs.pkl")
            with open(ppath2, "wb+") as f:
                pickle.dump(img_accumulator, f)
            del img_accumulator
            img_pickles[cam] = ppath2
    if(return_images):
        return cam_pickles, img_pickles
    return cam_pickles

def inside_dims(idx, mn, mx):
    return mn < idx <= mx    
def relate_tracks(dists, p1, p2):
    dists.argmin(axis=1)
    
    lims_cam2 = {}
    lim_low = 0

    voting = np.zeros_like(dists)
    voting[np.arange(0, dists.shape[0]),dists.argmin(axis=1)]=1
    
    trk_v_res = np.zeros((len(p1.keys()),len(p2.keys())))
    row_min = 0
    for i, k in enumerate(p1.keys()):
        row_mat = voting[row_min:row_min+len(p1[k]), :]
        col_min = 0
        for j, t in enumerate(p2.keys()):
            trk_v_res[i, j] = np.sum(row_mat[:,col_min:col_min+len(p2[t])])
            col_min += len(p2[t])
        row_min += len(p1[k])
    return trk_v_res
            
    
if __name__ == "__main__":
    in_path = "../datasets/AIC20_track3_MTMC/test/"
    out_path = "./out/cams"
    sequence = 3
    cameras = [10, 11, 12, 13, 14, 15]
    # cameras = [10, 11]
    fc_normalize = False
    load_pickles = True
    show_cars = True
    max_permitted_size = 150*150*3
    
    all_cam_dict = generate_track_for_all_cams(in_path,sequence,cameras)
    
    use_gpu = torch.cuda.is_available()

    dir_to_weights = '../../weights/resnet50_triple_10.pth' #Añadir la direccion als weights
    model = Model(dir_to_weights)
    # load_pretrained_weights(model, dir_to_weights)
    ppath = os.path.join(out_path, f"cam_pickles.pkl")
    ipath = os.path.join(out_path, f"cam_imgs_pickles.pkl")
    if not load_pickles:
        _r = generate_features(all_cam_dict,in_path,sequence,
                                                cameras, save_path=out_path,
                                                feature_func=model,
                                                return_images = show_cars,
                                                max_permitted_size=max_permitted_size)
        if(not show_cars):
            cam_pickles = _r
        else:
            cam_pickles, img_pickles = _r
            with open(ipath, "wb+") as f:
                pickle.dump(img_pickles, f)
        with open(ppath, "wb+") as f:
            pickle.dump(cam_pickles, f)
    else:
        # if(not show_cars):
        cam_pickles = {c: os.path.join(out_path, f"{c}.pkl") for c in cameras}
        # cam_pickles = pickle.load(open(ppath, "rb"))
        if(show_cars):
            # img_pickles = pickle.load(open(ipath, "rb"))
            img_pickles = {c: os.path.join(out_path, f"{c}_imgs.pkl") for c in cameras}

    
    relation_cams = calculate_matrices(cam_pickles)

    # if fc_normalize:
    #     feature_accumulated_norm = F.normalize(feature_accumulated, p=2, dim=1) #Cambiar para que solo 
    import display
    # display.display_heatmap(relation_cams[10][11])
    # a = "y"
    cv2.namedWindow("res")
    i = 11
    j = 10
    print(cam_pickles)
    p1 = pickle.load(open(cam_pickles[j], "rb"))
    if(show_cars):
        pc1 = pickle.load(open(img_pickles[j], "rb"))
        c1 = []
        for k in pc1.keys(): c1.extend(pc1[k]) 
    
    while  i < 16:
        # print(cam_pickles)
        p2 = pickle.load(open(cam_pickles[i], "rb"))
        if(show_cars):
            pc2 = pickle.load(open(img_pickles[i], "rb"))
            c2 = []
            for k in pc2.keys(): c2.extend(pc2[k]) 
        dists = relation_cams[j][i]
        trk_v_res = relate_tracks(dists, p1, p2)
    
        res = display.display_heatmap(dists)
        display.print_grid(res, p1, p2)
        # a = input("Continue? y/n")
        k = -1
        pressed = False
        if(show_cars):
            cv2.setMouseCallback("res", display.show_pair_imgs, (c1, c2))
        while not pressed:
            # rshow = display.obtain_img(res, display.sel_x, display.sel_y)
            cv2.imshow("res", res)
            x,y = display.obtain_xy()
            # print(x, y)
            try:
                img1 = c1[y]
                img2 = c2[x]
            except:
                pass
            if(x > -1 and y > -1):        
                cv2.imshow("car1", img1)
                cv2.imshow("car2", img2)
                res_i = display.print_axis(res)
                cv2.imshow("res", res_i)
            k = cv2.waitKey(1)
            if(k != -1):
                print("press 'c' to continue")
                print("key:", k)
            if(k == 99):
                i+=1
                pressed = True
    
    
    print()
    