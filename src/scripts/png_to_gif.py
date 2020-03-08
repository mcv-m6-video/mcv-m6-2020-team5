# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:04:47 2020

@author: hamdd
"""


import glob
from natsort import natsorted, ns
from PIL import Image

# Create the frames
frames = []
imgs = glob.glob("../plots/IoU_frame/*.png")
natsorted(imgs, key=lambda y: y.lower())
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('png_to_gif.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)

