'''
implementation of decision tree and random forest,based on the 
implementation of semantic texton forests
'''

from uti.computeInt import computeInt

def make_order(x1,x2):
    """make sure x1 > x2
    
    Arguments:
        x1 {[type]} -- [description]
        x2 {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    if x1 < x2:
        tmp = x2
        x2 = x1
        x1 = tmp
    return x1,x2

class node(object):
    def __init__(self,x1,y1,x2,y2,t,threshold):
        self.left = None
        self.right = None
        self.x1 = x1
        self.x2 = x2 
        self.y1 = y1 
        self.y2 = y2
        self.t = t 
        self.threshold = threshold

    def value_at_pixel(self,integral_im,i,j):

        x1, x2 = make_order(self.x1,self.x2)
        y1, y2 = make_order(self.y1,self.y2)

        hegiht,width,_ = integral_im.shape
        x1 += j
        x2 += j
        y1 += i
        y2 += i

        #checking boundary condition
        if x2 < 0: x2 = 0
        if y2 < 0: y2 = 0
        if x1 >= hegiht: x1 = hegiht-1
        if y1 >= width: y1 = width-1

        return integral_im[x1,y1] + integral_im[x2,y2] - integral_im[x1,y2] - integral_im[x2,y1]




class decision_tree(object):
    """class object decison tree. Function includes building a tree
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self,maxDepth,numFeature,numThreshold,labelWeights):

        self.maxDepth = maxDepth
        self.numFeature = numFeature
        self.numThreshold = numThreshold
        self.labelWeights = labelWeights
        self.root = None

    def computeBestThresh(self):
        pass



    def computeDepthFirst(self,int_ims,gt_ims):
        pass





class random_forest(object):
    def __init__(self):
        pass

