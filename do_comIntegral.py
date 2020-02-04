'''
main file to compute integral images

'''
import numpy as np
import os 
from tqdm import trange
from tqdm import tqdm
from numba import njit
from config import config
from uti.computeInt import computeInt
import h5py
import time
import pdb


def computeInt(textons):
    """compute integral image ii  
    
    Arguments:
        textons {list of np.array} -- list of images, of size (height,width,depth)
    """
    lis = []
    # for each image 
    for l in trange(len(textons)):
        im = textons[l]
        height, width, depth = im.shape
        ii = np.zeros((height+1,width,depth))
        for j in range(1,height+1):
            r = np.zeros(depth) #tmp row sum
            for i in range(width):
                # handling boundary case
                r = r + im[j-1,i,:]
                ii[j,i,:] = ii[j-1,i,:] + r
        lis.append(ii[1:,:,:])

    return lis




#@njit
def combine_textons(all_textons,texton_offset,subsample=1):
    """combine different types of textons together as one data
    
    Arguments:
        all_textons {list of list of np.array} -- [[im1,im2,im3,...],[im1,im2,im3,...],[]], each sublist is a type of textons
        texton_offset {[type]} -- [description]
        subsample {int} -- only compute 
    Return:
        list of integral images
    """
    ims_textons_combine = []
    nims = len(all_textons[0])
    ntextons = len(texton_offset)
    print('starting computing for each image!')
    # for each image
    for im_id in tqdm(range(nims)):
        height, width = all_textons[0][im_id].shape
        # subsample
        nh = (height-1)//subsample + 1
        nw = (width-1)//subsample + 1
        kk = texton_offset[-1]

        res = np.zeros((int(nh),int(nw),int(kk))) 
        for texton_id in range(ntextons):
            dat = all_textons[texton_id][im_id]
            for j in range(height):
                for i in range(width):
                    if texton_id == 0:
                        res[int(j//subsample),int(i//subsample),int(dat[j,i])] = 1
                    else:
                        res[int(j//subsample),int(i//subsample),int(texton_offset[texton_id-1] + dat[j,i])] = 1

        ims_textons_combine.append(res)
    
    return ims_textons_combine


if __name__ == '__main__':

    #=============
    # train 
    #=============

    ntextons = len(config['texton_type']) # types of textons
    texton_offset = np.zeros(ntextons)
    all_textons = []
    l = 0

    # read in different types of textons and compute texton-offset
    for texton in config['texton_type']:

        single_texton = [] #store one type of textons
        texton_path = os.path.join(config['texton_path'],'train_msrc_' + texton + '.h5')
        f = h5py.File(texton_path,'r')
        grp_name = list(f.keys())[0]
        ims_names = list(f[grp_name].keys()) # store image name for later use

        for name in f[grp_name].keys():
            single_texton.append(f[grp_name][name][:])

        all_textons.append(single_texton)

        for ele in single_texton:
            tmp_max = np.max(ele)
            if texton_offset[l] < tmp_max:
                texton_offset[l] = tmp_max + 1

        l += 1

        f.close()

    for i in range(1,ntextons):
        texton_offset[i] += texton_offset[i-1]

    train_texton_combine = combine_textons(all_textons,texton_offset)

    # calculate ii_ims
    print('starting computing integral images')
    ii_ims = computeInt(train_texton_combine) 
    print('finishing computing integral images')   
     
    # save
    tmp = 'train_'
    for texton in config['texton_type']:
        tmp = tmp + texton + '_'

    save_path = os.path.join(config['intIms_path'],tmp + '.h5')
    f = h5py.File(save_path,'w')
    grp = f.create_group('list of ii_ims')
    l = 0

    for l in trange(len(ims_names)):
        grp.create_dataset(ims_names[l],data=ii_ims[l],compression='gzip',compression_opts=9)
        
    f.close()


    
    


    

    


        


