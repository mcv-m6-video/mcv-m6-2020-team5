#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:04:25 2020

@author: dazmer
"""
import numpy as np




def obtain_votation_mat(dists, p1, p2):
    dists.argmin(axis=1)

    voting = np.zeros_like(dists)
    voting[np.arange(0, dists.shape[0]),dists.argmin(axis=1)]=1
    
    trk_v_res = np.zeros((len(p1.keys()),len(p2.keys())))
    trk_v_size = np.zeros((len(p1.keys()),len(p2.keys())))
    row_min = 0
    for i, k in enumerate(p1.keys()):
        row_mat = voting[row_min:row_min+len(p1[k]), :]
        col_min = 0
        for j, t in enumerate(p2.keys()):
            trk_v_res[i, j] = np.sum(row_mat[:,col_min:col_min+len(p2[t])])
            trk_v_size[i, j] = len(p1[k])
            col_min += len(p2[t])
        row_min += len(p1[k])
        
    winner_idx = np.argmax(trk_v_res,axis=1)
    winner_n_votes = np.max(trk_v_res,axis=1)
    win_percent = winner_n_votes/trk_v_size[:,0]
    return winner_idx, win_percent

def relate_tracks(dists, p1, p2, win_thrs=0.3):
    winner_idx, win_percent = obtain_votation_mat(dist, p1, p2)
    
    translate_dict_p1 = {}
    translate_dict_p2 = {}
    current_id = 0
    for i,val in enumerate(winner_idx):
        if(win_percent[i] > win_thrs):
            
            translate_dict_p1[list(p1.keys())[i]] = current_id
            translate_dict_p2[list(p2.keys())[val]] = current_id
            current_id += 1
        else:
            translate_dict_p1[list(p1.keys())[i]] = current_id
            current_id += 1
    
    for k in p2.keys():
        if(k not in translate_dict_p2):
            translate_dict_p2[k] = current_id
            current_id += 1
        
    return translate_dict_p1,translate_dict_p2

def relate_tracks_two_cams(dist, p1, p2, win_thrs=0.3):
    winner_idx, win_percent = obtain_votation_mat(dist, p1, p2)
    
    trans = {}
    for i,val in enumerate(winner_idx):
        k1 = list(p1.keys())[i]
        k2 = list(p2.keys())[val]
        if(win_percent[i] > win_thrs):
            trans[k1] = k2
    unassigned_c1 = [k for k in p1.keys() if k not in trans.keys()]
    unassigned_c2 = [k for k in p2.keys() if k not in trans.values()]
        
    return trans, unassigned_c1, unassigned_c2

def create_tracklet(mat_relations, cameras, id_c1, trk_id_c1):
    idx_c1 = np.where(np.array(cameras)==id_c1)[0]
    tracklet = np.array([np.ones(len(cameras))*-1])
    tracklet[0, idx_c1] = trk_id_c1
    mat_relations = np.vstack((mat_relations,tracklet))
    return mat_relations

def generate_mat(mat_relations, cam_names, 
                c1_name, c2_name, 
                trk_id_c1, trk_id_c2):
    
    # We select camera's rows
    idx_c1 = np.where(np.array(cam_names)==c1_name)[0][0]
    idx_c2 = np.where(np.array(cam_names)==c2_name)[0][0]
    
    # Then, we select where does the tracklet id appears on the first camera column
    rows_idxs = np.where(mat_relations[:, idx_c1]==trk_id_c1)[0]
    
    # If there is no tracklet, we initialize it
    # And vertically stack it on mat_relations
    if(len(rows_idxs)==0):
        # tracklet = np.ones(len(cam_names))*-1
        # tracklet[idx_c1] = trk_id_c1
        # mat_relations = np.vstack((mat_relations,tracklet))
        mat_relations = create_tracklet(mat_relations, cam_names, 
                                                c1_name, trk_id_c1)
        rows_idxs = [-1]
    
    # So, for every occurence of r_idx, we assign the new item
    for r_idx in rows_idxs:
        tracklet = mat_relations[r_idx, :]
        if(tracklet[idx_c2] == -1):
            tracklet[idx_c2] = trk_id_c2
    return mat_relations


    
def generate_global_dict(mat_relations, cameras, id_c1, id_c2, p1, p2, dists, **kwargs):
    trans, unassigned_c1, unassigned_c2 = relate_tracks_two_cams(dists, 
                                                                 p1, p2, 
                                                                 win_thrs=0.3)
    
    for k_c1, k_c2 in trans.items():
        mat_relations = generate_mat(mat_relations, cameras, id_c1, id_c2, 
                                     k_c1, k_c2)
        
    idx_c1 = np.where(np.array(cameras)==id_c1)[0][0]
    for k_c1 in unassigned_c1:
        rows_idxs = np.where(mat_relations[:, idx_c1]==k_c1)[0]
        if(not len(rows_idxs)):
            mat_relations = create_tracklet(mat_relations, cameras, 
                                                 id_c1, k_c1)
            
    idx_c2 = np.where(np.array(cameras)==id_c2)[0][0]
    for k_c2 in unassigned_c2:
        rows_idxs = np.where(mat_relations[:, idx_c2]==k_c2)[0]
        if(not len(rows_idxs)):
            mat_relations = create_tracklet(mat_relations, cameras, 
                                                    id_c2, k_c2)
        
    return mat_relations
        