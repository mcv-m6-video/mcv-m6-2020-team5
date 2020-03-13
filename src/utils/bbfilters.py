#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This file contains a set of filters for Bounding Boxes analyisis.

Created on Wed Oct 31 12:19:00 2018

@author: daniaz
"""
import cv2
import numpy as np

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	if len(cnts):
        	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def bbfilters(rects,im, wf_high=0.95, hf_high=0.8, wf_low=0.02, hf_low=0.1, 
    min_size=0.001, max_size=0.1, form_factor_low=0.3, form_factor_high=10, debug=False,
    reverse = True):
    """
    @brief Filters large, small, and uncommon bboxes filling them with white
    @param im image that has to be cleared
    @param org image to be filled with ones and be returned
    @param wf_high maximum width of the bbox relative to im width size
    @param wf_low  minimum width of the bbox relative to im width size
    @param hf_high maximum height of the bbox relative to im height size
    @param hf_low  minimum height of the bbox relative to im height size
    @param min_size minimum pixel size for the bbox relative to image's pixels
    @param form_factor_low minimum bbox ratio (w/h)
    @param form_factor_high maximum bbox ratio (w/h)
    @param debug wether or not show debug comments (no waitKey)
    @return image image cleared
    @return new_bb_list new list of unfiltered bboxes
    """
    ih,iw = im.shape[:2]
    wf_high = iw*wf_high if isinstance(wf_high, float) else wf_high
    hf_high = hf_high*ih if isinstance(hf_high, float) else hf_high
    wf_low  = max(iw*wf_low, 3) if isinstance(wf_low, float) else wf_low 
    hf_low  = max(ih*hf_low, 3) if isinstance(hf_low, float) else hf_low
    min_size = im.size*min_size if isinstance(min_size, float) else min_size     
    max_size = im.size*max_size if isinstance(max_size, float) else max_size     
    
    good_bboxes = []
    bad_bboxes = []
    # print("==============")
    for x1,y1,x2,y2 in rects:
        w = x2-x1
        h = y2-y1
        box_factor = float(w)/float(h)
        box_size = w*h
        # print("-----------")
        # msg = f"h:{h}, max:{hf_high} min:{hf_low}"
        # msg += f"\nw:{w}, max:{wf_high} min:{wf_low}"
        # msg += f"\nFF:{box_factor} max:{form_factor_high}, min:{form_factor_low}"
        # msg +=f"\n bsz:{box_size} min:{min_size}, max:{max_size}"
        # print(msg)
        if(w > wf_high or h > hf_high or w < wf_low or h < hf_low or \
        box_factor > form_factor_high or box_factor < form_factor_low or \
        min_size > box_size or max_size < box_size):
            bad_bboxes.append((x1,y1,x2,y2))
        else:
            good_bboxes.append((x1,y1,x2,y2))

    return good_bboxes, bad_bboxes
    
def bbfilters_filling(im, org=None, wf_high=0.95, hf_high=0.8, wf_low=0.02, hf_low=0.1, 
    min_size=0.001, form_factor_low=0.3, form_factor_high=10, debug=False,
    reverse = True):
    """
    @brief Filters large, small, and uncommon bboxes filling them with white
    @param im image that has to be cleared
    @param org image to be filled with ones and be returned
    @param wf_high maximum width of the bbox relative to im width size
    @param wf_low  minimum width of the bbox relative to im width size
    @param hf_high maximum height of the bbox relative to im height size
    @param hf_low  minimum height of the bbox relative to im height size
    @param min_size minimum pixel size for the bbox relative to image's pixels
    @param form_factor_low minimum bbox ratio (w/h)
    @param form_factor_high maximum bbox ratio (w/h)
    @param debug wether or not show debug comments (no waitKey)
    @return image image cleared
    @return new_bb_list new list of unfiltered bboxes
    """
    ih,iw = im.shape[:2]
    wf_high = iw*wf_high if isinstance(wf_high, float) else wf_high
    hf_high = hf_high*ih if isinstance(hf_high, float) else hf_high
    wf_low  = max(iw*wf_low, 3) if isinstance(wf_low, float) else wf_low 
    hf_low  = max(ih*hf_low, 3) if isinstance(hf_low, float) else hf_low
    min_size = im.size*min_size if isinstance(min_size, float) else min_size     
    
    max_val = np.max(im)
    r_im = max_val-im if reverse else im
    r_im = r_im.astype(np.uint8)
    cnts,_ = cv2.findContours(r_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, bb_list = sort_contours(cnts)   
    if(org is not None): 
        image=org.copy()
    else:
        image=im.copy()
    if(org is not None): im = org
    new_bb_list = []
    
    
    for x,y,w,h in bb_list:
        box_factor = float(w)/float(h)
        box_size = w*h
        if(w > wf_high or h > hf_high or w < wf_low or h < hf_low or \
        box_factor > form_factor_high or box_factor < form_factor_low or \
        min_size > box_size):
            image[y:y+h, x:x+w] = np.ones((h,w))*max_val
        else:
            new_bb_list.append((x,y,w,h))

    for x,y,w,h in new_bb_list:
        
        image[y:y+h, x:x+w] = im[y:y+h, x:x+w]
    
    if debug:
        rect = im.copy()
        for x,y,w,h in bb_list:
            rect = cv2.rectangle(rect,(x,y),(x+w,y+h),(0,0,0),2)
        cv2.imshow("Contours", rect)
        cv2.imshow("BBFilters",image)
    
#    if(reverse):
#        image = max_val-image
    return image, new_bb_list