import glob 

import os
import random
import os

from dataset_analysis import dict_classes



def split_dataset(path):
    # l_img = glob.glob(input_dir+"/*")
    dict_classes
    print("Number of images found:", len(l_img))
    
    s_img = random.sample(l_img, k=int(len(l_img)*0.8))
    print(f"Proceed redoing {len(s_img)} instances?")
    a = input("Continue? y/n")
    if (a == "y"):
        for pimg in s_img:
            print("Removing", pimg)
            # os.remove(pimg)
    else:
        print("not doing anything:", a)
    
if __name__ == "__main__":
    input_dir = "/media/dazmer/datasets/traffic/AIC20_track3_MTMC/aic20_reID_bboxes/bounding_box_test"
    split_dataset(input_dir)