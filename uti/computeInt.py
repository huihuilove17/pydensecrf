import numpy as np
from numba import njit




@njit
def computeInt(textons):
    """compute integral image ii  
    
    Arguments:
        textons {list of np.array} -- list of images, of size (height,width,depth)
    """
    lis = []
    # iterate through images
    for l in range(len(textons)):
        im = textons[l]
        height, width, depth = im.shape
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

