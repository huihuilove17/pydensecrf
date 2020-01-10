'''
implement texton data and textonboost classifier

'''

class textonData(object):
    """class object for each pixel, should have the following information
        1 position x and y
        2 texton id
        3 class id

    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self,i,j,c):
        """initialize some variables
        
        Arguments:
            object {[type]} -- [description]
            i {int} -- [image col id]
            j {int} -- [image row id]
        """
        self.j_ =  j
        self.i_ = i
        self.class = c 
        
    def evalInt(self,x1,y1,x2,y2,t):
        """computer 
        
        Arguments:
            x1 {[type]} -- [description]
            y1 {[type]} -- [description]
            x2 {[type]} -- [description]
            y2 {[type]} -- [description]
            t {[type]} -- [description]
        """
        pass


class joinBoost(object):
    pass


class textonboost(object):
    """class object for textonboost calssifier
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(object):
        pass

    def random_pick