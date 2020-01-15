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
        kernelx = np.array([[0,0,0],[1,0,-1],[0,0,0]])
        kernely = np.array([[0,-1,0],[0,0,0],[0,1,0]])

        # compute x and y gradients
        gx = fftconvolve(dat,kernelx,'same')
        gy = fftconvolve(dat,kernely,'same')
        g = np.sqrt(gx ** 2 + gy ** 2) #magnitude
        theta = np.degree(np.arctan(gy/(gx + le-05)))%180 #direction in unsigned degree

        



