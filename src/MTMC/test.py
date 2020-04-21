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
from metrics import calculate_matrices, recalculate_matrices
from collections import OrderedDict

import glob
from mot import mot_metrics
import track_relation
# from track_relation import generate_global_dict
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

def generate_track_for_all_cams(in_path, sequence_num, camera_list, method):
    
    all_cam_dict = {}
    seq_name = "S{:02d}".format(sequence_num)
    in_path = os.path.join(in_path,seq_name)
    for cam in camera_list:
        cam_name = "c{:03d}".format(cam)
        annot_path = os.path.join(in_path,cam_name)
    
        # temp_dict = read_gt(annot_path)
        # all_cam_dict[cam] = regenerate_tracks(temp_dict)
        if method == 'det':
            all_cam_dict[cam] = read_det(annot_path)
        elif method == 'gt':
            all_cam_dict[cam] = read_gt(annot_path)
        
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


def arrange_dict_by_id(trans_dict,p1):
    new_dict = {}
    for k in p1.keys():
        new_k = trans_dict[k]
        new_dict[new_k] = p1[k]
    return new_dict

def reid_from_dict(trans_dict,cam_dict):
    new_dict = {}
    for frame in cam_dict.keys():
        frame_boxes = []
        boxes = cam_dict[frame]
        for box in boxes:
            new_k = trans_dict[box[4]]
            frame_boxes.append((box[0],box[1],box[2],box[3],new_k,box[5]))
            
        new_dict[frame] = frame_boxes
    return new_dict

def reid_form_matrix(matrix, all_cam_dict):
    all_cam_dict_new = {}
    for id_c in all_cam_dict.keys():
        idx_c = np.where(np.array(cameras)==id_c)[0][0]
        cam_dict = all_cam_dict[id_c]
        new_dict = {}
        for frame in cam_dict:
            frame_boxes = []
            boxes = cam_dict[frame]
            for box in boxes:
                res = np.where(matrix[:, idx_c]==box[4])
                new_k = res[0][0]
                frame_boxes.append((box[0],box[1],box[2],box[3],new_k,box[5]))
            new_dict[frame] = frame_boxes
        all_cam_dict_new[id_c] = new_dict
    return all_cam_dict_new

                
def read_number_frames(path, camera):
    with open (path, 'rt') as number_frames:
        for line in number_frames:
            if line.split(' ')[0] == "c" + f"{camera:03d}":   
                total_frames = int(line.split(' ')[1])  
    return total_frames  

def merge_cameras_dict(dict1, dict2): 
    ''' Merge dictionaries and keep values of common keys in list'''
    # dict3 = {**dict1, **dict2}
    # for key, value in dict3.items():
    #     if key in dict1 and key in dict2:
    #         dict3[key] = value.extend(dict1[key])
            
    dict3 = {}
    
    for key in dict1.keys():
        if(key in dict2.keys()):
            dict3[key] = dict1[key] + dict2[key]
        else:
            dict3[key] = dict1[key]
    
    for key in dict2.keys():
        if(key not in dict3.keys()):
            dict3[key] = dict2[key]
 
    return dict3

# def merge_cameras_matrix(cam_matrix, dict1, dict2, id_c1, id_c2):
#     dict3 = dict1.copy()
#     idx_c1 = np.where(np.array(cameras)==id_c1)[0][0]
#     idx_c2 = np.where(np.array(cameras)==id_c2)[0][0]
    
    
#     for tracklet in cam_matrix:
#         idx_trk_c1 = tracklet[idx_c1]
#         idx_trk_c2 = tracklet[idx_c2]
#         if(idx_trk_c1 == -1 and idx_trk_c2 == -1):
#             continue
#         if(idx_trk_c1 != -1 and idx_trk_c2 == -1):
#             dict3[idx_trk_c1] = dict1[idx_trk_c1]
#         if(idx_trk_c1 == -1 and idx_trk_c2 != -1):
#             dict3[idx_trk_c2] = dict2[idx_trk_c2]
#         if(idx_trk_c1 != -1 and idx_trk_c2 != -1):
#             dict3[idx_trk_c1]
#     for k in dict2.keys():
#         dict3[key] = 
    
def evaluate_mot(mot_obj,gt_dict,pred_dict,num_frames):
    
    for frame in range(num_frames):
        if frame in pred_dict:
            dt_rects = pred_dict[frame]
        else:
            dt_rects = []   
        if frame in gt_dict:
            gt_rects = gt_dict[frame]  
        else:
            gt_rects = []          
        mot_obj.update(dt_rects,gt_rects)
    
def evaluate_mot2(cam_list,mot_obj,gt_dict,pred_dict,num_frames):
    max_frames = max(num_frames.values())
    
    for i in range(max_frames):
        for cam_num in cam_list:
            if(i < num_frames[cam_num]):
                if i in pred_dict[cam_num]:
                    dt_rects = pred_dict[cam_num][i]
                else:
                    dt_rects = []   
                if i in gt_dict[cam_num]:
                    gt_rects = gt_dict[cam_num][i]  
                else:
                    gt_rects = []          
                mot_obj.update(dt_rects,gt_rects)

    
if __name__ == "__main__":
    in_path = "../../datasets/AIC20_track3_MTMC/test/"
    out_path = "./out/cams"
    sequence = 3
    cameras = [10, 11, 12, 13, 14, 15]
    # cameras = [10, 11]
    fc_normalize = False
    load_pickles = True
    show_cars = True
    max_permitted_size = 150*150*3
    use_matrix = True
    merge_features = False
    number_frames = {}
    view_validation = False
    win_thr = 0.3
    visualize_votation = True

    
    for cam in cameras:
        number_frames[cam] = read_number_frames("../../datasets/AIC20_track3_MTMC/cam_framenum/S" + f"{sequence:02d}" + '.txt', cam)

    all_cam_dict = generate_track_for_all_cams(in_path,sequence,cameras,'det')
    gt_all_cam_dict = generate_track_for_all_cams(in_path,sequence,cameras,'gt')
    
    use_gpu = torch.cuda.is_available()

    # name = "osnet_ain_x1_0"
    # dir_to_weights = "../../weights/osnet_ain_x1_0.pth"
    name = "resnet50"
    dir_to_weights = '../../weights/resnet50_triple_old.pth' #Añadir la direccion als weights
    model = Model(name, dir_to_weights)
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
        cam_pickles = OrderedDict()
        for c in cameras:
            cam_pickles[c] = os.path.join(out_path, f"{c}.pkl")
            # cam_pickles = pickle.load(open(ppath, "rb"))
            
        if(show_cars):
            # img_pickles = pickle.load(open(ipath, "rb"))
            img_pickles = OrderedDict()
            for c in cameras:
                img_pickles[c] = os.path.join(out_path, f"{c}_imgs.pkl")

    
    relation_cams = calculate_matrices(cam_pickles)

    # if fc_normalize:
    #     feature_accumulated_norm = F.normalize(feature_accumulated, p=2, dim=1) #Cambiar para que solo 
    import display
    # display.display_heatmap(relation_cams[10][11])
    # a = "y"
    cv2.namedWindow("res")
    # i = 11
    # j = 10
    cam_merge = {}

    
    for j in [13]:
        # print(cam_pickles)
        p1 = pickle.load(open(cam_pickles[j], "rb"))
        if(show_cars):
            pc1 = pickle.load(open(img_pickles[j], "rb"))
            c1 = []
            for k in pc1.keys(): c1.extend(pc1[k]) 
        mat_relations = np.empty((0, len(cameras)))
        for i in cameras:
            if i==j: continue
            # print(cam_pickles)
            p2 = pickle.load(open(cam_pickles[i], "rb"))
            if(merge_features):
                if not cam_merge: 
                    pass
                else: 
                    relation_cams = recalculate_matrices(cam_merge, j, p2, i)
                    p1 = cam_merge
                    
            if(show_cars):
                pc2 = pickle.load(open(img_pickles[i], "rb"))
                c2 = []
                for k in pc2.keys(): c2.extend(pc2[k]) 
            dists = relation_cams[j][i]
    
            if(not use_matrix):
                translate_dict_p1,translate_dict_p2 = track_relation.relate_tracks(dists, p1, p2, win_thrs=win_thr)
                for b in range(10,i):
                    all_cam_dict[b] = reid_from_dict(translate_dict_p1,all_cam_dict[b])
                all_cam_dict[i] = reid_from_dict(translate_dict_p2,all_cam_dict[i]) 
                
                id_p1 = arrange_dict_by_id(translate_dict_p1, p1)
                id_p2 = arrange_dict_by_id(translate_dict_p2, p2)
            else:
                mat_relations = track_relation.generate_global_dict(mat_relations, cameras, 
                                                     j, i, p1, p2, dists,
                                                     win_thrs=win_thr)
                # for b in range(10,i):
                #     all_cam_dict[b] = reid_from_matrix(mat_relations,all_cam_dict[b])
                # all_cam_dict[i] = reid_from_dict(translate_dict_p2,all_cam_dict[i]) 
                print(mat_relations)    
            # generate_mat(mat_relations, cameras, j, i, 
    
            
    
            if(merge_features):
                if(not use_matrix):
                    cam_merge = merge_cameras_dict(id_p1, id_p2) 
                else:
                    raise(NotImplementedError("Merge is not available for matrix yet"))
    
            
            if(visualize_votation):
                res = display.display_heatmap(dists)
                display.print_grid(res, p1, p2)
                # a = input("Continue? y/n")
                k = -1
                pressed = False
                if(show_cars):
                    cv2.setMouseCallback("res", display.show_pair_imgs, (c1, c2))
                print(img_pickles[j], img_pickles[i], j, i, len(c1), len(c2))
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
                        pressed = True
        tracking_metrics = mot_metrics()            
        
        if(use_matrix):
            all_cam_dict = reid_form_matrix(mat_relations, all_cam_dict)
        for cam_num in cameras:
            evaluate_mot(tracking_metrics,gt_all_cam_dict[cam_num],
                         all_cam_dict[cam_num],number_frames[cam_num])
            idf1, idp, idr,precision,recall = tracking_metrics.get_metrics()
            # print(f"Cam {cam_num} has idf1: ", idf1)
        # evaluate_mot2(cameras,tracking_metrics,gt_all_cam_dict,all_cam_dict,number_frames)
     
        # tracking_metrics.update(new_p2,gt_all_cam_dict[i])
        idf1, idp, idr, precision, recall  = tracking_metrics.get_metrics()
        print("idf1: ", idf1)
        print("idp: ", idp)
        print("idr: ", idr)
        print("precision: ", precision)
        print("recall: ", recall)
        
        print()
        
        if(view_validation):
            display.view_cars_matrix(mat_relations, img_pickles)
    