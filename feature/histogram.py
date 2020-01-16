'''
python file for histogram of gradient
'''
import numpy
from scipy.signal import fftconvolve


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

    def evaluate_an_image(self,lab_im,level):
        """compute feature response at different level
        
        Arguments:
            lab_im {[type]} -- [description]
            level {[type]} -- [description]
        """
        
        dat = lab_im[:,:,level]
        height. width = dat.shape
        kernelx = np.array([[0,0,0],[1,0,-1],[0,0,0]])
        kernely = np.array([[0,-1,0],[0,0,0],[0,1,0]])
        nAnglebins = 9
        cellSize = 5 # for simplicity, we assume odd number
        ncells = 3

        # compute x and y gradients
        gx = fftconvolve(dat,kernelx,'same')
        gy = fftconvolve(dat,kernely,'same')
        g = np.sqrt(gx ** 2 + gy ** 2) #magnitude
        theta = np.degree(np.arctan(gy/(gx + le-05)))%180 #direction in unsigned degree

        # construct for each pixel
        res = np.zeros(height,width,nAnglebins)
        for j in range(height):
            for i in range(width):
                # linear interpolation
                l1 = theta[j,i] // 20, l2 = l1 + 1
                a1 = theta[j,i] - l1 * 20
                a2 = l2 * 20 -theta[j,i]
                if l2 >= nAnglebins:
                    l2 = 0
                res[j,i,l1] = g[j,i]*a2/(a1+a2)
                res[j,i,l2] = g[j,i]*a1/(a1+a2)
            
        # every 5*5 pixels form a cell, each ele in cellBin is average magnitude for ith angle of a cell  
        cellBin = [] 
        
        kernelx = np.zeros((5,5)); kernelx[2,:] = [1,1,1,1,1]
        kernely = np.zeros((5,5)); kernely[:,2] = [1,1,1,1,1]

        for k in range(nAnglebins):
            dat = res[:,:,k]
            cellBin.append(fftconvolve(fftconvolve(dat,kernelx,'same'),kernely,'same')/25)
        
        # finalize hog feature
        # 3 * 3 cells form a block, for each pixel, feature vector is of the size nAnglebins * ncells * ncells
        total_size = nAnglebins * ncells * ncells
        res = np.zeros()


        
        


            




        # every 3*3 cels forms a block
        



