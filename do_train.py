'''
main file for training(already have textons)
remain to add validation data
'''

import numpy as np
from uti.loadtextons import loadtextons
from uti.Image import loadLabelImage
from uti.computeInt import computeInt
from numba import njit

#config

texton_type = ['filterbank','color','location']
train_texton_path = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/texton/train'
test_texton_path = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/texton/test'
train_files = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/Train.txt'
test_files = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/Test.txt'



if __name__ == '__main__':

    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    # load training textons
    with open(train_files,'r') as f1:
        train_names = f1.readlines()

    train_textons = loadtextons(train_texton_path,texton_type)    # list of images(size: height, width, depth)
    train_labels = loadLabelImage(train_names) #list of images(size: height,width)

    # compute integral images
    print('start computing integral image')
    textonsInt = computeInt(train_textons)

    print('finishing!')

    np.save('/Users/huihuibullet/Documents/project/pydensecrf-1/ii_ims.npy',textonsInt)
    np.save('/Users/huihuibullet/Documents/project/pydensecrf-1/gt_ims.npy',train_labels)   

    # build random forest


    # save random forest

    np.load = np_load_old
    