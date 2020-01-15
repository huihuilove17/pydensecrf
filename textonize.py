'''
Textonize all the images 
'''
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.io import imshow
from skimage.color import rgb2lab
from feature.filterbank import FilterBank
import os
from sklearn.cluster import KMeans
from uti.loadimages import loadimages
import random



if __name__ == '__main__':

    train_path = '/home/hanhui/Documents/pydensecrf/data/Train.txt'
    test_path  = '/home/hanhui/Documents/pydensecrf/data/Test.txt'
    train_names = []
    test_names = []

    with open(train_path,'r') as fi:
        image_files = fi.readlines()

    train_names = [ele.strip('\n') for ele in image_files][:3]

    with open(test_path,'r') as fi:
        image_files = fi.readlines()

    test_names = [ele.strip('\n') for ele in image_files][:3]

    texton = Texton()
    feature = FilterBank(5)
    nTextons = 400
    texton.fit_train(train_names,feature,nTextons)
    texton.evaluate(test_names)
    texton.saveTextons('/home/hanhui/Documents/pydensecrf/texton')




