'''
main file for training(already have textons)
'''

import numpy as np
from uti.loadtextons import loadtextons
from uti.Image import loadLabelImage
from numba import njit

#config

texton_type = ['filterbank','color','location']
train_texton_path = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/texton/train'
test_texton_path = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/texton/test'
train_files = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/Train.txt'
test_files = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/Test.txt'



if __name__ == '__main__':

    # load training textons
    with open(train_files,'r') as f1:
        train_names = f1.readlines()

    train_textons = loadtextons(train_texton_path,texton_type)    # list of images(size: height, width, depth)
    train_labels = loadLabelImage(train_names) #list of images(size: height,width)

    # compute integral images
        



    # build random forest


    # save random forest