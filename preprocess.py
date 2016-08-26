"""
Preprocesses training and testing data.
"""
import cv2
import numpy as np

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols),
    	dtype=np.uint8)

    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows),
        	interpolation=cv2.INTER_CUBIC)
    return imgs_p