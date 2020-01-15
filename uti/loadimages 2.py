'''
mapping color of gt_images to a specific id
'''
import numpy as np
import os
from skimage.io import imread
from skimage.color import rgb2lab


def init_msrc():
    """construct a color id map for ground truth images
    """
    color_to_id = dict()
    color_to_id[np.array([128,0,0]).tobytes()] = 0
    color_to_id[np.array([0,128,0]).tobytes()] = 1
    color_to_id[np.array([128,128,0]).tobytes()] = 2 
    color_to_id[np.array([0,0,128]).tobytes()] = 3
    color_to_id[np.array([0,128,128]).tobytes()] = 4
    color_to_id[np.array([128,128,128]).tobytes()] = 5
    color_to_id[np.array([192,0,0]).tobytes()] = 6
    color_to_id[np.array([64,128,0]).tobytes()] = 7
    color_to_id[np.array([192,128,0]).tobytes()] = 8
    color_to_id[np.array([64,0,128]).tobytes()] = 9
    color_to_id[np.array([192,0,128]).tobytes()] = 10
    color_to_id[np.array([64,128,128]).tobytes()] = 11
    color_to_id[np.array([192,128,128]).tobytes()] = 12
    color_to_id[np.array([0,64,0]).tobytes()] = 13
    color_to_id[np.array([128,64,0]).tobytes()] = 14
    color_to_id[np.array([0,192,0]).tobytes()] = 15
    color_to_id[np.array([128,64,128]).tobytes()] = 16
    color_to_id[np.array([0,192,128]).tobytes()] = 17
    color_to_id[np.array([128,192,128]).tobytes()] = 18
    color_to_id[np.array([64,64,0]).tobytes()] = 19
    color_to_id[np.array([192,64,0]).tobytes()] = 20
    color_to_id[np.array([0,0,0]).tobytes()] = -1
    color_to_id[np.array([64,0,0]).tobytes()] = -2
    color_to_id[np.array([128,0,128]).tobytes()] = -3

    return color_to_id

def id_to_color():
    id_to_color = dict()
    id_to_color[0] = np.array([128,0,0]) 
    id_to_color[1] = np.array([0,128,0]) 
    id_to_color[2] = np.array([128,128,0]) 
    id_to_color[3] = np.array([0,0,128]) 
    id_to_color[4] = np.array([0,128,128]) 
    id_to_color[5] = np.array([128,128,128]) 
    id_to_color[6] = np.array([192,0,0]) 
    id_to_color[7] = np.array([64,128,0]) 
    id_to_color[8] = np.array([192,128,0]) 
    id_to_color[9] = np.array([64,0,128]) 
    id_to_color[10] = np.array([192,0,128]) 
    id_to_color[11] = np.array([64,128,128]) 
    id_to_color[12] = np.array([192,128,128]) 
    id_to_color[13] = np.array([0,64,0]) 
    id_to_color[14] = np.array([128,64,0]) 
    id_to_color[15] = np.array([0,192,0]) 
    id_to_color[16] = np.array([128,64,128]) 
    id_to_color[17] = np.array([0,192,128]) 
    id_to_color[18] = np.array([128,192,128]) 
    id_to_color[19] = np.array([64,64,0]) 
    id_to_color[20] = np.array([192,64,0]) 
    id_to_color[-1] = np.array([0,0,0]) 
    id_to_color[-2] = np.array([64,0,0]) 
    id_to_color[-3] = np.array([128,0,128]) 

    return id_to_color



def loadimages(file_names):
    """loading images
    
    Arguments:
        file_names {list of strs} -- list of file names
    
    Keyword Arguments:
        type {str} -- [description] (default: {'train'})
    """
    ims = []
    lab_ims = []
    gt_ims = []
    id_ims = [] 
    colorID = init_msrc()
    idColor = id_to_color()
    for ele in file_names:
        image_path = os.path.join('/home/hanhui/Documents/pydensecrf/data/msrc/Images',ele)
        #gt_path = os.path.join('/home/hanhui/Documents/pydensecrf/data/msrc/GroundTruth',ele)
        im = imread(image_path)
        lab_im = rgb2lab(im)
        #gt_im = imread(gt_path)
        #id_im = np.zeros((im.shape[0],im.shape[1])) 
        # map gt_im's pixels to id
        # convert to id image
        '''
        for j in range(gt_im.shape[0]):
            for i in range(gt_im.shape[1]):
                tmp = gt_im[j,i,:]
                if tmp in colorID:
                    id_im[j,i] = colorID[tmp]
                else:
                    id_im[j,i] = 0
        '''

        ims.append(im)
        lab_ims.append(lab_im)
        #gt_ims.append(gt_im)
        #id_ims.append(id_im)

    #return ims, lab_ims, gt_ims, id_ims, idColor,file_names
    return ims,lab_ims




