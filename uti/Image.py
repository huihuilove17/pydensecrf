'''
convert rgb image to lab image
'''

import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab
from matplotlib import pyplot as plt
import os


def init_msrc():
    """construct a color id map for ground truth images
    """
    color_to_id = dict()
    color_to_id[tuple(np.array([128,0,0]))] = 0
    color_to_id[tuple(np.array([0,128,0]))] = 1
    color_to_id[tuple(np.array([128,128,0]))] = 2 
    color_to_id[tuple(np.array([0,0,128]))] = 3
    color_to_id[tuple(np.array([0,128,128]))] = 4
    color_to_id[tuple(np.array([128,128,128]))] = 5
    color_to_id[tuple(np.array([192,0,0]))] = 6
    color_to_id[tuple(np.array([64,128,0]))] = 7
    color_to_id[tuple(np.array([192,128,0]))] = 8
    color_to_id[tuple(np.array([64,0,128]))] = 9
    color_to_id[tuple(np.array([192,0,128]))] = 10
    color_to_id[tuple(np.array([64,128,128]))] = 11
    color_to_id[tuple(np.array([192,128,128]))] = 12
    color_to_id[tuple(np.array([0,64,0]))] = 13
    color_to_id[tuple(np.array([128,64,0]))] = 14
    color_to_id[tuple(np.array([0,192,0]))] = 15
    color_to_id[tuple(np.array([128,64,128]))] = 16
    color_to_id[tuple(np.array([0,192,128]))] = 17
    color_to_id[tuple(np.array([128,192,128]))] = 18
    color_to_id[tuple(np.array([64,64,0]))] = 19
    color_to_id[tuple(np.array([192,64,0]))] = 20
    color_to_id[tuple(np.array([0,0,0]))] = -1
    color_to_id[tuple(np.array([64,0,0]))] = -2
    color_to_id[tuple(np.array([128,0,128]))] = -3

    return color_to_id




def loadImages(names):

    """loading images
    
    Arguments:
        names {list of str} -- list of image names
    
    Returns:
        [type] -- [description]
    """
    res = []
    for ele in names:
        image_path = os.path.join('/Users/huihuibullet/Documents/project/pydensecrf-1/data/msrc/Images',ele)
        im = imread(image_path)
        res.append(im)

    return res

def rgb2labs(ims):
    """convert rgb image to lab images
    
    Arguments:
        ims {list of class object Image} -- [description]
    
    Returns:
        [list of class object Image] -- [description]
    """
    res = []
    for im in ims:
        lab_im = rgb2lab(im)
        res.append(lab_im)

    return res


def loadLabelImage(names):
    """loading ground truth image of size height * width 
    
    Arguments:
        names {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    gts = []
    color_map = init_msrc()

    for name in names:
        str = name.split('.')[0]+'_GT.bmp'
        image_path = os.path.join('/Users/huihuibullet/Documents/project/pydensecrf-1/data/msrc/GroundTruth',str)
        im = imread(image_path)
        height, width, _ = im.shape
        gt_im = np.zeros((height,width))

        for j in range(height):
            for i in range(width):
                val = color_map[tuple(im[j,i,:])]
                gt_im[j,i] = val
        gts.append(gt_im)

    return gts
    






