'''
implement texton data and textonboost classifier

'''
import numpy as np

def computeII(x,y,t):
    """compute ii(x,y) 
    
    Arguments:
        x {[type]} -- [description]
        y {[type]} -- [description]
    """






class textonData(object):
    """class object for each pixel, should have the following information
        1 position x and y
        2 texton id
        3 class id

    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self,i,j):
        """initialize some variables
        
        Arguments:
            object {[type]} -- [description]
            i {int} -- [image col id]
            j {int} -- [image row id]
        """
        self.j_ =  j
        self.i_ = i
        
    def evalInt(self,image, x1,y1,x2,y2,t):
        """compute integral image. 
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

        hegiht,width = image.shape
        #checking boundary condition
        if x2 < 0: x2 = 0
        if y2 < 0: y2 = 0
        if x1 >= hegiht: x1 = hegiht
        if y1 >= width: y1 = width

        # compute region ul and ur
        ii1 = np.zeros(2)
        flag = 0
        for j in range(x2+1):
            s_tmp = 0
            # for each col
            for i in range(y1+1):
                val = 1 if image[j,i] == t else 0
                s_tmp += val
                if i == y2:
                    ii1[flag] += s_tmp
                    flag = 1
                elif i == y1:
                    ii1[flag] += s_tmp
                    flag = 0
        

        # compute region ll and lr
        ii2 = np.zeros(2)
        flag = 0
        for j in range(x2+1,x1+1):
            s_tmp = 0
            # for each col
            for i in range(y1+1):
                val = 1 if image[j,i] == t else 0
                s_tmp += val
                if i == y2:
                    ii2[flag] += s_tmp
                    flag = 1
                elif i == y1:
                    ii2[flag] += s_tmp
                    flag = 0
        
        return ii2[1] + ii1[0] - ii2[0] - ii1[1]

                    










class joinBoost(object):
    pass





class textonboost(object):
    """class object for textonboost calssifier
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self):
        pass

    def random_pick(self):
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

 