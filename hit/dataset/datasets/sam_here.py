import numpy as np
import torch
import matplotlib.pyplot as plt 
import cv2
import os
from tqdm import tqdm

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "/home/josmy/Code/HIT/data/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

orig_dir = "/data1/jhmdb/jhmdb/videos/"
dest_dir = "/data1/jhmdb/jhmdb/videos_new/"

def show_anns(anns, image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    all_ims = []
    for ann in sorted_anns:
        if ann["stability_score"] > 0.9:
            m = ann['segmentation']
            im = image * (np.stack([m, m, m], axis=2))
            all_ims.append(im)
        
    img = sum(all_ims)
    return img

for d in tqdm(os.listdir(orig_dir)):
    dest_v = dest_dir + d
    if not os.path.exists(dest_v):
        os.makedirs(dest_v)
    for f in os.listdir(orig_dir + d):
        image = cv2.imread(orig_dir + d + "/" + f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        masks = mask_generator.generate(image)
        img = show_anns(masks, image)
        
        cv2.imwrite(dest_v + "/" + f, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        


        
    
# image = cv2.imread('00002.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



# print(len(masks))
# print(masks[0]["bbox"])
# print(masks[0]["segmentation"].shape)
# print(masks[-1])

# im = 
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# print(img.shape)
# print(type(img))
# cv2.imwrite("00002_mod.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))