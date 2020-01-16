'''
Compute location feature?
'''

import numpy as np
from numba import jit


class Location(object):
    def __init__(self):
        self.size_ = 2
        self.name_ = 'location'

    def get_size(self):
        return self.size_

    def get_name(self):
        return self.name_

    def evaluate_an_image(self,lab_im):
        
        height,width,_ = lab_im.shape
        res = np.zeros((height,width,2))

        for j in range(height):
            for i in range(width):
                res[j,i,0] = j/(height-1)
                res[j,i,1] = i/(width-1)
        return res


if __name__ == "__main__":
    
    file = '~/Documents/pydensecrf/data/msrc/Images/8_30_s.bmp'
    image = imread(file)

    hog = location()
    res = hog.evaluate_an_image(image)

 