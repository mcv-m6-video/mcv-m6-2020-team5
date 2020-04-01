import cv2
import numpy as np
from dataset_video import dataset_video


def get_starting_video_offsets(filename):
    
    offsets = []
    with open(filename,"r") as offset_file:
        lines = offset_file.readlines()
        for line in lines:
            cam_name, cam_offset = line.split()
            cam_offset = cam_offset.replace(".","") # numbers have the thousand dot
            offsets.append(int(cam_offset))
    return offsets


def main():
    
    video_path_list = []
    video_path_list.append("../datasets/AIC20_track3_MTMC/AIC20_track3/test/S02/c006/vdo.avi")
    video_path_list.append("../datasets/AIC20_track3_MTMC/AIC20_track3/test/S02/c007/vdo.avi")
    video_path_list.append("../datasets/AIC20_track3_MTMC/AIC20_track3/test/S02/c008/vdo.avi")
    video_path_list.append("../datasets/AIC20_track3_MTMC/AIC20_track3/test/S02/c009/vdo.avi")
    
    video_offset_list = np.array(get_starting_video_offsets("../datasets/AIC20_track3_MTMC/AIC20_track3/cam_timestamp/S03.txt"))
    
    video_offset_list = video_offset_list[:len(video_path_list)]
    minval = np.max(video_offset_list[1:])
    video_offset_list = minval - video_offset_list
    
    video_list = []
    for path,offset in zip(video_path_list,video_offset_list):
        video_list.append(dataset_video(path,offset_ms=offset))
    
    images_correct = True
    frame_idx = 0
    while(all([vid.video_capture.isOpened() for vid in video_list]) and images_correct):
        print(frame_idx)
        frame_idx += 1
        frames = []
        images_correct = True
        for vid in video_list:
            ret, frame = vid.video_capture.read()
            frames.append(frame)
            images_correct *= ret
        
        if(images_correct):
            for i,frame in enumerate(frames):
                frame = cv2.resize(frame,(480,270))
                cv2.imshow(str(i),frame)
            cv2.waitKey(100) 

    



if __name__ == "__main__":
    main()