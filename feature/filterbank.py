'''
Convolution filter bank
'''
#from skimage.io import imread
#from skimage.io import imshow
#import matplotlib.pyplot as plt

import numpy as np
import math
from scipy.signal import fftconvolve

pi = 3.14159265358979323846

#=================================================================
#define kenel
#define 1d gaussian kernel vector
def GaussianKernel(sigma):
    halfsize = int(np.ceil(3*sigma))
    kernel = []
    s2 = sigma * sigma
    for i in np.arange(-halfsize,halfsize+1):
        tmp = 1/np.sqrt(2*pi*s2)*np.exp(-i**2/(2*s2))
        kernel.append(tmp)

    res = np.zeros((2*halfsize+1,2*halfsize+1))
    res[halfsize,:] = kernel
    return res

#define 1d gaussian derivative kernel vector
def GaussianDerivativeKernel(sigma):
    halfsize = int(np.ceil(3*sigma))
    kernel = []
    s2 = sigma * sigma
    for i in np.arange(-halfsize,halfsize+1):
        tmp = 1/np.sqrt(2*pi*s2)*np.exp(-i**2/(2*s2))*(-i/s2)
        kernel.append(tmp)
    res = np.zeros((2*halfsize+1,2*halfsize+1))
    res[halfsize,:] = kernel
    
    return res 

#define 1d gaussian laplacian kernel vector
def GaussianLaplacian(sigma):
    halfsize = int(np.ceil(3*sigma))
    kernel = []
    s2 = sigma * sigma
    f = 1/np.sqrt(2*pi*s2)
    w = 1/s2
    w2 = 1/(2*s2)
    for i in np.arange(-halfsize,halfsize+1):
        tmp = (i*i*w*w-w)*f*np.exp(-i*i*w2)
        kernel.append(tmp)
    res = np.zeros((2*halfsize+1,2*halfsize+1))
    res[halfsize,:] = kernel
 
    return res 


#=================================================================
#define operation
# assume input kernel is of the form [[0,0,0],[-1,0,1],[0,0,0]] 
#@profile
def convolutionX(image,kernel,filling):

    return fftconvolve(image,kernel,'same')


def convolutionY(image,kernel):

    return fftconvolve(image,kernel.T,'same')

def convolutionXY(image,kernelX,kernelY):
    
    res = convolutionX(image,kernelX)
    res1 = convolutionY(res,kernelY)
    return res1


def convoLog(image,kernel1,kernel2):
    r1 = convolutionX(convolutionY(image,kernel1),kernel2)
    r2 = convolutionX(convolutionY(image,kernel2),kernel1)

    return r1 + r2


# getting the channel
def getChannel(lab_image,c):
    
    return lab_image[:,:,c] 

#=================================================================

# define class object 
class FilterBank(object):
    # initialize kernels
    def __init__(self,kappa):
        self.g1 = GaussianKernel(1*kappa)
        self.g2 = GaussianKernel(2*kappa)
        self.g3 = GaussianKernel(3*kappa)
        self.g4 = GaussianKernel(4*kappa)
        self.dg1 = GaussianDerivativeKernel(2*kappa)
        self.dg2 = GaussianDerivativeKernel(4*kappa)
        self.lg1 = GaussianLaplacian(1*kappa)
        self.lg2 = GaussianLaplacian(2*kappa)
        self.lg3 = GaussianLaplacian(4*kappa)
        self.lg4 = GaussianLaplacian(8*kappa)
        self.size_ = 17
        self.name_ = 'filterbank'
    
    def getSize(self):
        return self.size_

    # use kernprof to monitor the time 
    #@profile
    def evaluate_an_image(self,lab_image):
        """compute feature response for a single image 
        
        Arguments:
            lab_image {[type]} -- [description]
        
        Returns:
            [np.array] -- [of size (M,H,W), where M is dimension of the feature response]
        """
        # get the image for L,a,b channel in lab_image
        L = getChannel(lab_image,0)
        a = getChannel(lab_image,1)
        b = getChannel(lab_image,2)

        
        i1 = convolutionXY(L,self.g1,self.g1)
        i2 = convolutionXY(L,self.g2,self.g2)
        i3 = convolutionXY(L,self.g3,self.g3)

        i4 = convolutionXY(a,self.g1,self.g1)
        i5 = convolutionXY(a,self.g2,self.g2)
        i6 = convolutionXY(a,self.g3,self.g3)

        i7 = convolutionXY(b,self.g1,self.g1)
        i8 = convolutionXY(b,self.g2,self.g2)
        i9 = convolutionXY(b,self.g3,self.g3)

        i10 = convolutionX(convolutionY(L,self.dg1),self.g2)
        i11 = convolutionX(convolutionY(L,self.dg2),self.g4)
        i12 = convolutionY(convolutionX(L,self.dg1),self.g2)
        i13 = convolutionY(convolutionX(L,self.dg2),self.g4)

        i14 = convoLog(L,self.lg1,self.g1)
        i15 = convoLog(L,self.lg2,self.g2)
        i16 = convoLog(L,self.lg3,self.g3)
        i17 = convoLog(L,self.lg4,self.g4)

        print('finishing convolution!')
        return np.array((i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17))
    
    def get_name(self):
        return self.name_
    

# testing

if __name__ == "__main__":
    
    file = '~/Documents/pydensecrf/data/msrc/Images/8_30_s.bmp'
    image = imread(file)

    g1 = GaussianKernel(3)
    dg1 = GaussianDerivativeKernel(2*1)

    height, width = image1.shape[0], image1.shape[1]  

    L = getChannel(image1,0)
    a = getChannel(image1,1)
    b = getChannel(image1,2)

    tmp = FilterBank(15)

    '''
    #res = tmp.evaluate_an_image(image1)
    kernel1 = np.arange(100).reshape((10,10))
    kernel2 = np.arange(225).reshape((15,15))
    convolutionX(a,kernel1)
    convolutionX(L,kernel2)
    '''
    res = tmp.evaluate_an_image(image1)
    









        