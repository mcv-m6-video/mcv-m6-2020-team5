# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:03:07 2020

@author: Group 5
"""
import cv2

SOURCE = "../datasets/AICity_data/train/S03/c010/vdo.avi"
def main():
    cap = cv2.VideoCapture(SOURCE)
    ret, frame = cap.read()
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow('Frame',frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    cap.release()
    
    
if __name__ == "__main__":
    main()
    