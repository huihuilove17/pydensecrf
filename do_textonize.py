'''
Main file to textonize all the images 
'''
import os
from uti.texton import Texton
from feature.filterbank import FilterBank
from feature.color import Color
from feature.location import Location
from feature.histogram import HoG


if __name__ == '__main__':

    # setting 
    train_path = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/Train.txt'
    test_path  = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/Test.txt'
    save_path = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/texton'
    
    # get training and testing names
    train_names = []
    test_names = []

    with open(train_path,'r') as fi:
        image_files = fi.readlines()

    train_names = [ele.strip('\n') for ele in image_files][:3]

    with open(test_path,'r') as fi:
        image_files = fi.readlines()

    test_names = [ele.strip('\n') for ele in image_files][:3]

    # training 
    print('start training!')
    feature = HoG()
    nTextons = 400
    
    texton = Texton(feature)
    texton.fit_(train_names,nTextons,200)
    texton.saveTextons(save_path)   
    print('finish training!')
    
    
    # testing
    print('start testing') 
    texton.evaluate(test_names)
    texton.saveTextons(save_path,mode='test')
    print('finish testing')




