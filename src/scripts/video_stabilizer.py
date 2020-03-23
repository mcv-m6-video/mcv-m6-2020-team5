from vidstab import VidStab
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

def main(original_video_file, stabilized_video_file):

    stabilizer = VidStab()
    stabilizer.stabilize(input_path=original_video_file, output_path=stabilized_video_file)

    stabilizer.plot_trajectory()
    plt.show()

    stabilizer.plot_transforms()
    plt.show()
    
def save_video(original_video_file, stabilized_video_file, combined_video_file):
    cap_orig = cv2.VideoCapture(original_video_file)
    cap_stab = cv2.VideoCapture(stabilized_video_file)
    i = 0
    while(cap_orig.isOpened()):
        ret_orig, frame_orig = cap_orig.read()
        ret_stab, frame_stab = cap_stab.read()
        if i == 0:
            height, width, layers = frame_orig.shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # define the video codec
            video = cv2.VideoWriter(combined_video_file, fourcc, 30, (width, height))
        if ret_orig is True and ret_stab is True:
            frame_orig = cv2.resize(frame_orig, (int(width/2), height))
            frame_stab = cv2.resize(frame_stab, (int(width/2), height))
            vis = np.concatenate((frame_orig, frame_stab), axis=1)
            video.write(vis)
            i += 1
            print('Frame: ', i)
        else:
            break
        
    
if __name__ == "__main__":
    input_file='../datasets/videos/input3.mp4' 
    output_file='../datasets/videos/vistab3.avi'
    combined_video = '../datasets/videos/combined3.avi'
    main(input_file, output_file)
    save_video(input_file, output_file,combined_video)
    
