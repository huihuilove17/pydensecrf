'''
python file for histogram of gradient
'''
import numpy as np
from numba import jit 
from scipy.signal import fftconvolve
# testing
from skimage.io import imread
import time
import pdb


class HoG(object):
    """class object for histogram of oriented gradients
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self):
        self.size_ = None
        self.name_ = 'hog'

    def get_name(self):
        return self.name_

    def get_size(self):
        return self.size_

    def evaluate_an_image(self,lab_im,level):
        """compute feature response at different level
        
        Arguments:
            lab_im {[type]} -- [description]
            level {[type]} -- [description]
        """
        dat = lab_im[:,:,level]
        height, width = dat.shape
        kernelx = np.array([[0,0,0],[1,0,-1],[0,0,0]])
        kernely = np.array([[0,-1,0],[0,0,0],[0,1,0]])
        nAngleBins = 9
        ncells = 3
        cellSize = 5
        total_size = nAngleBins * ncells * ncells
        self.size_ = total_size
        
        # compute x and y gradients
        gx = fftconvolve(dat,kernelx,'same')
        gy = fftconvolve(dat,kernely,'same')
        g = np.sqrt(gx ** 2 + gy ** 2) #magnitude
        theta = np.degrees(np.arctan(gy/(gx + 0.0000001)))%180 #direction in unsigned degree

        # construct for each pixel
        res = np.zeros((height,width,nAngleBins))
        for j in range(height):
            for i in range(width):
                # linear interpolation
                l1 = int(theta[j,i] // 20)
                l2 = l1 + 1
                a1 = theta[j,i] - l1 * 20
                a2 = l2 * 20 -theta[j,i]
                if l1 >= nAngleBins:
                    l1 = 0
                    l2 = 1
                elif l2 >= nAngleBins:
                    l2 = 0
                res[j,i,l1] = g[j,i]*a2/(a1+a2)
                res[j,i,l2] = g[j,i]*a1/(a1+a2)
            
        # every 5*5 pixels form a cell, each ele in cellBin is average magnitude for ith angle of a cell  
        cellBin = [] 
        
        kernelx = np.zeros((5,5)); kernelx[2,:] = [1,1,1,1,1]
        kernely = np.zeros((5,5)); kernely[:,2] = [1,1,1,1,1]

        for k in range(nAngleBins):
            dat = res[:,:,k]
            tmp = fftconvolve(fftconvolve(dat,kernelx,'same'),kernely,'same')/25
            cellBin.append(tmp.astype(np.float64))
        
        # finalize hog feature
        # 3 * 3 cells form a block, for each pixel, feature vector is of the size nAnglebins * ncells * ncells
        cellBin = np.array(cellBin)
        @jit(nopython=True)
        def finalize():
            res = np.zeros((height,width,total_size))
            for j in range(height):
                for i in range(width):
                    cid = 0
                    # iterate through 9 cells
                    for jj in np.linspace(j-cellSize,j+cellSize,ncells):
                        for ii in np.linspace(i-cellSize,i+cellSize,ncells):
                            # checking boundary condition
                            if jj < 0: jj = 0
                            if jj >= height: jj = height-1
                            if ii < 0: ii = 0
                            if ii >= width: ii = width-1
                            res[j,i,cid*nAngleBins:(cid+1)*nAngleBins] += cellBin[:,int(jj),int(ii)]
                            cid += 1
        
            return res

        res = finalize()
        return res


if __name__ == "__main__":
    
    file = '~/Documents/pydensecrf/data/msrc/Images/8_30_s.bmp'
    image = imread(file)

    hog = HoG()
    res = hog.evaluate_an_image(image,1)

    









        


                






        



