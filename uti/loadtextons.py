'''
loading textons, organize data like
    /data
        /texton
            /train
                msrc_filterbank.npy
                msrc_color.npy
                ...
            /test
                msrc_filterbank.npy
                msrc_color.npy
                
'''
import numpy as np
import os 
from numba import njit


texton_type = ['filterbank','color','location']


def loadtextons(path,texton_type,subsample = 1):
    """loading all types of textons
    
    Arguments:
        path {str} -- path of texton files, like /data/texton/train
        texton_type {list of strs} -- list of texton names

    """
    ntextons = len(texton_type) # types of textons
    texton_offset = np.zeros(ntextons)
    textons = []
    l = 0
    # read in different types of textons
    for texton in texton_type:
        texton_path = os.path.join(path,'msrc_' + texton + '.npy')
        dat = np.load(texton_path) # list of np.array texton_map (height,width)
        textons.append(dat)

        # for each image, find the possible max texton id
        for id in range(len(dat)):
            tmp = np.max(dat[id])
            if texton_offset[l] < tmp:
                texton_offset[l] = tmp + 1
        l += 1

    '''
    if we have three types of textons, filterbank, color and location, each has classes 400, 10 and 20
    then texton_offset will be 400 10,20 
    '''
    for i in range(1,ntextons):
        texton_offset[i] += texton_offset[i-1]

    
    # sampling 
    
    def finalize(textons):
        """ using jit to speed up
        
        Arguments:
            textons {list of list of np.array} -- [description]
        
        Returns:
            [type] -- [description]
        """

        # combine all the textons together
        
        ims_textons_combine = []
        # iterate through each type of textons
        for texton_id in range(ntextons):
            for texton_map in textons[texton_id]:
                # texton map is for single image 
                height, width = texton_map.shape 

                # subsampleing 
                nh = (height-1)//subsample + 1
                nw = (width-1)//subsample + 1
                kk = texton_offset[-1]
                res = np.zeros((int(nh),int(nw),int(kk)))

                for j in range(height):
                    for i in range(width):
                        if texton_id == 0:
                            res[int(j//subsample),int(i//subsample),int(texton_map[j,i])] = 1
                        else:
                            res[int(j//subsample),int(i//subsample),int(texton_offset[texton_id-1] + texton_map[j,i])] = 1
                
                ims_textons_combine.append(res) # list of textons_combines for each image 
        
        return ims_textons_combine
    
    res = finalize(textons)
    
    return res


if __name__ == '__main__':

    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    path = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/texton/train'
    res = loadtextons(path,texton_type)

    print(res[0].shape)

    np.load = np_load_old

    

    


        


