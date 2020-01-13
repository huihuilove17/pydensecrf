'''
implement texton data and textonboost classifier

'''
import numpy as np
import random

def gauss(stddev):
    """following the original c code
    
    Arguments:
        stddev {[type]} -- [description]
    """
    while True:
        u = np.random.random(0,1) * 2-1
        v = np.random.random(0,1) * 2-1
        w = u**2 + v**2
        if w < 1:
            break
    
    w = np.sqrt(-2*np.log(w)/w)
    return u * w * stddev

def gaussRange(stddev2):
    stddev = stddev2 / 2
    return stddev + gauss(stddev)
            


class textonData(object):
    """ each image has a textonData class object, joinboost algorithm uses a list of textonData 
    for training
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self,image,i,j):
        """initialize some variables
        
        Arguments:
            object {[type]} -- [description]
            i {int} -- [image col id]
            j {int} -- [image row id]
        """
        self.j_ =  j
        self.i_ = i
        self.image = image
        
    
    def value(self,x1,y1,x2,y2,t):
        """compute v[r,t](i)
           let x stands for rows(vertical direction), y stands for cols(horizontal direction)
           assume x1,y1 is the below right corner of the rectangle
                  x2, y2 is the upper left corner 
        
        Arguments:
            x1 {[type]} -- [description]
            y1 {[type]} -- [description]
            x2 {[type]} -- [description]
            y2 {[type]} -- [description]
            t {[type]} -- [description]

        """
        #making right order
        if x2 > x1:
            tmp = x1
            x1 = x2
            x2 = tmp
        if y2 > y1:
            tmp = y1
            y1 = y2 
            y2 = tmp

        hegiht,width,_ = self.image.shape
        x1 += self.j_
        x2 += self.j_
        y1 += self.i_
        y2 += self.i_

        #checking boundary condition
        if x2 < 0: x2 = 0
        if y2 < 0: y2 = 0
        if x1 >= hegiht: x1 = hegiht
        if y1 >= width: y1 = width

        tmp = image[x2:x1+1,y2:y1+1] == t
        tmp *= 1
        return np.sum(tmp)



class textonClassifier(object):
    """class object for textonboost calssifier(waek classifier)
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self):
        self.threshold = None
        self.min_recSize = 5
        self.max_recSIze = 100
        self.x1 = None 
        self.x2 = None
        self.y1 = None
        self.y2 = None
        
        
    def set_threshold(self,t)
        self.threshold = t

    def random_pick(self):
        """randomly pick a region 
        """
        # determine width and height for the rectangle
        w = np.random.uniform(self.min_recSize,self.max_recSIze)
        h = np.random.uniform(self.min_recSize,self.max_recSIze)

        # randomly pick a point
        # gaussian range(not knowing why?)
        x = gaussRange(self.max_recSIze-h)
        y = gaussRange(self.max_recSIze-w)


        self.x2 = x - self.max_recSIze/2
        self.y2 = y - self.max_recSIze/2
        self.x1 = x2 + h
        self.y1 = y2 + w

        



class textonboost(object):
    def __init__(self):
        pass





if __name__ == "__main__":

    image = np.array([[1,0,1,1],[0,1,0,0],[1,0,1,1],[0,1,2,1]])
    x1,y1,x2,y2 = 3,3,0,0

    # compute region ll and lr
    ii2 = np.zeros(2)
    flag = 0 if y2 > 1 else 1
    for j in range(x2,x1+1):
        s_tmp = 0
        # for each col
        for i in range(y1+1):
            val = 1 if image[j,i] == 1 else 0
            s_tmp += val
            # handling boundary case
            if i == y2-1:
                ii2[flag] += s_tmp
                flag = 1
            elif i == y1:
                ii2[flag] += s_tmp
                flag = 0

    print(ii2)
    print(ii2[1] - ii2[0]) 

 