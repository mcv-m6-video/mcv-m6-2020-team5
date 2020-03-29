import cv2
import numpy as np

class dataset_video(object):
    def __init__(self, video_path, offset_ms=0):
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        self.offset_ms = offset_ms
        
        if(offset_ms != 0):
            self.video_capture.set(cv2.CAP_PROP_POS_MSEC,offset_ms)
        
        
    def reload_video(self):
        self.video_capture = cv2.VideoCapture(self.video_path)
        