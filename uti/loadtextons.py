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

#texton_type = ['filterbank','color','location']

def loadtextons(path,texton_type,subsample = 1):
    """loading all types of textons
    
    Arguments:
        path {str} -- path of texton files, like /data/texton/train
        texton_type {list of strs} -- list of texton names

    """
    ntextons = len(texton_type) # types of textons
    texton_offset = np.zeros(ntextons)
    textons = []

    # read in different types of textons
    for texton in texton_type:
        texton_path = os.path.join(path,'msrc_' + texton + '.npy')
        dat = np.load(texton_path)
        textons.append(dat)

    n_ims,height,width = textons[0].shape # texton_id is of type <n_ims,height,width>
    
    # calculate texton offset
    for l in range(ntextons):
        texton = textons[l] # of type <ntrain * height * width>
        
        # for each image, find the possible max texton id
        for id in range(n_ims):
            tmp = np.max(texton[id])
            if texton_offset[l] < tmp:
                texton_offset[l] = tmp + 1

    '''
    if we have three types of textons, filterbank, color and location, each has classes 400, 10 and 20
    then texton_offset will be 400 10,20 
    '''
    for i in range(1,ntextons):
        texton_offset[i] += texton_offset[i-1]

    @njit
    def finalize(textons):
        # combine all the textons together
        ims_textons_combine = []
        nh = (height-1)//subsample + 1
        nw = (width-1)//subsample + 1
        kk = texton_offset[-1]

        # for each image 
        for l in range(n_ims):
            res = np.zeros((int(nh),int(nw),int(kk)))
            # for each type of textons
            for k in range(ntextons):
                tmp1 = textons[k]
                tmp = tmp1[l]
                for j in range(height):
                    for i in range(width):
                        if k == 0:
                            res[int(j//subsample),int(i//subsample),int(tmp[j,i])] = 1
                        else:
                             res[int(j//subsample),int(i//subsample),int(texton_offset[k-1] + tmp[j,i])] = 1

            ims_textons_combine.append(res) # ims_textons_combine is a list
        
        return ims_textons_combine
    
    res = finalize(textons)
    
    return res


if __name__ == '__main__':
    path = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/texton/train'
    res = loadtextons(path,texton_type)

    print(res[0].shape)

    

    


        


