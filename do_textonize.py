'''
Main file to textonize all the images 
'''
import os
from uti.texton import Texton
from feature.filterbank import FilterBank
from feature.color import Color
from feature.location import Location
from feature.histogram import HoG
from config import config


 
if __name__ == '__main__':

    # get names
    train_names = []
    test_names = []
    valid_names = []

    with open(config['train_path'],'r') as fi:
        image_files = fi.readlines()

    train_names = [ele.strip('\n') for ele in image_files][:3]

    with open(config['test_path'],'r') as fi:
        image_files = fi.readlines()

    test_names = [ele.strip('\n') for ele in image_files][:3]

    with open(config['valid_path'],'r') as fi:
        image_files = fi.readlines()

    valid_names = [ele.strip('\n') for ele in image_files][:3]

    save_path = config['texton_path']

    '''
    #==================
    #filterbank
    #==================

    print('start training filterbank!')
    feature = FilterBank(config['kappa'])
    texton = Texton(feature)
    texton.train_kmeans(train_names,config['filter_nTextons'],config['samples_per_image'],ifLAB=True)
    print('finishing training filterbank')

    
    # train 
    print('start computing train images')
    texton.evaluate(train_names,mode='train') 
    texton.saveTextons(save_path,mode='train')
    print('finish computing train images') 

    # valid
    print('start computing valid images')
    texton.evaluate(valid_names,mode='valid') 
    texton.saveTextons(save_path,mode='valid')
    print('finish computing valid images') 

    
    # test
    print('start computing test images')
    texton.evaluate(test_names,mode='test') 
    texton.saveTextons(save_path,mode='test')
    print('finish computing test images') 

    #==================
    #color
    #==================

    print('start training color!')
    feature = Color()
    texton = Texton(feature)
    texton.train_kmeans(train_names,config['color_nTextons'],config['samples_per_image'],ifLAB=True)
    print('finishing training color')

    
    # train 
    print('start computing train images')
    texton.evaluate(train_names,mode='train') 
    texton.saveTextons(save_path,mode='train')
    print('finish computing train images') 

    # valid
    print('start computing valid images')
    texton.evaluate(valid_names,mode='valid') 
    texton.saveTextons(save_path,mode='valid')
    print('finish computing valid images') 

    
    # test
    print('start computing test images')
    texton.evaluate(test_names,mode='test') 
    texton.saveTextons(save_path,mode='test')
    print('finish computing test images') 

    #==================
    #location
    #==================
    # training 
    
    print('start training location!')
    feature = Location() 
    texton = Texton(feature)
    texton.train_kmeans(train_names,config['location_nTextons'],config['samples_per_image'],ifLAB=True)
    print('finishing training location')

    
    # train 
    print('start computing train images')
    texton.evaluate(train_names,mode='train') 
    texton.saveTextons(save_path,mode='train')
    print('finish computing train images') 

    # valid
    print('start computing valid images')
    texton.evaluate(valid_names,mode='valid') 
    texton.saveTextons(save_path,mode='valid')
    print('finish computing valid images') 

    
    # test
    print('start computing test images')
    texton.evaluate(test_names,mode='test') 
    texton.saveTextons(save_path,mode='test')
    print('finish computing test images') 

    '''
    
    #==================
    # HoG
    #==================
    print('start training HoG!')
    feature = HoG(config['level'])
    texton = Texton(feature)
    texton.train_kmeans(train_names,config['hog_nTextons'],config['samples_per_image'],ifLAB=True)
    print('finishing training HoG')

    # train 
    print('start computing train images')
    texton.evaluate(train_names,mode='train') 
    texton.saveTextons(save_path,mode='train')
    print('finish computing train images') 

    # valid
    print('start computing valid images')
    texton.evaluate(valid_names,mode='valid') 
    texton.saveTextons(save_path,mode='valid')
    print('finish computing valid images') 

    # test
    print('start computing test images')
    texton.evaluate(test_names,mode='test') 
    texton.saveTextons(save_path,mode='test')
    print('finish computing test images') 

