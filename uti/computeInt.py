'''
function to compute integral images
'''

import numpy as np
from numba import njit

@njit
def computeInt(textons):
    """compute integral image ii  
    
    Arguments:
        textons {list of np.array} -- list of images, of size (height,width,depth)
    """
    height,width,depth = textons[0].shape
    lis = []
    # iterate through images
    for l in range(len(textons)):
        im = textons[l]
        ii = np.zeros((height,width,depth))

        # iterate at full resolution
        for k in range(depth):
            for j in range(height):
                r = 0
                for i in range(width):
                    # handleing boundary case
                    if j == 0: tmp1 = 0 
                    else: tmp1 = ii[j-1,i,k]

                    r = r + im[j,i,k]
                    ii[j,i,k] = tmp1 + r

        lis.append(ii)
    return lis


if __name__ == '__main__':
    data = np.zeros([3,3,2])
    data[:,:,0] = np.array([[1,0,1],[0,1,1],[1,1,1]])
    data[:,:,1] = np.array([[1,0,0],[0,0,1],[0,1,0]])

    res = computeInt([data])
    print(res[0][:,:,0])
    print(res[0][:,:,1])
    