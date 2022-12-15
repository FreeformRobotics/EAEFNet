import os

import cv2
import numpy as np
from PIL import Image

def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette


if __name__ == '__main__':
    pic_list = os.listdir("./")
    for i in range(len(pic_list)):
        img_cv = cv2.imread("./"+pic_list[i],-1)
        palette = get_palette()
        img = np.zeros((img_cv.shape[0], img_cv.shape[1], 3), dtype=np.uint8)
        for cid in range(0, len(palette)): # fix the mistake from the MFNet code on Dec.27, 2019
            img[img_cv == cid] = palette[cid]
        img = Image.fromarray(np.uint8(img))
        img.save("./"+"i"+pic_list[i])
        img.show()



