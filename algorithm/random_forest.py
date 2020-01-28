'''
implementation of decision tree and random forest,based on the 
implementation of semantic texton forests
'''
import numpy as np
import random 
import math
from uti.computeInt import computeInt

#=====================
#helper functon
#=====================

            

#=====================
# class object 
#=====================

class Node(object):
    def __init__(self,h,w,offset,t):
        self.left = None
        self.right = None
        self.thresh = None
        self.h = h
        self.w = w
        self.offset = offset
        self.t = t 
        
        
    def value_at_pixel(self,ii_im,i,j):
        """evalute the feature response at position (i,j)
        
        Arguments:
            ii_im {[type]} -- [description]
            i {[type]} -- [description]
            j {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """

        hegiht,width,_ = ii_im.shape

        x2 = j + self.offset
        y2 = i + self.offset
        if x2 < 0: x2 = 0
        if y2 < 0: y2 = 0

        x1 = x2 + self.h
        y1 = y2 + self.w

       if x1 >= hegiht: x1 = hegiht-1
       if y1 >= width: y1 = width-1

       return ii_im[x1,y1] + ii_im[x2,y2] - ii_im[x1,y2] - ii_im[x2,y1]

    def set_threshold(self,threshold):
        self.thresh = threshold


class decision_tree(object):
    """class object decison tree. Function includes building a tree
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self,maxDepth,numFeature,numThreshold,labelWeights,minRecSize, maxRecSize,num_class):

        self.maxDepth = maxDepth
        self.numFeature = numFeature
        self.numThreshold = numThreshold
        self.labelWeights = labelWeights
        self.max_recSize = maxRecSize
        self.min_recSize = minRecSize
        self.num_class = num_class
        self.root = None


    def computeBestThresh(self,values,labels,node):
        """find the optimal threshold for the current node
        
        Arguments:
            values {[type]} -- [description]
            labels {list of int} -- follows the order in the values
            node {[type]} -- [description]
        """
        candidates = np.randn(self.numThreshold) * np.std(values) + np.mean(values)
        info_gain = np.ones(self.numThreshold) * -99999999 # should be replace by MIN_INT
        
        # compute info gain for each threshold
        # for left and right branch, calculate shannon entropy for each class seperately
        for i in range(self.numThreshold):
            
            lid = np.where(values < candidates[i])[0] # id of left brance
            rid = np.where(values >= candidates[i])[0] # id of right brance
            gain = 0
            ltmp = np.zeros(self.num_class)
            rtmp = np.zeros(self.num_class)

            N = len(values)
            NL = len(lid) + le-4
            NR = len(rid) + le-4

            for j in range(len(ltmp)):
                ltmp[j] = np.sum((labels[lid] == j) * 1)
                rtmp[j] = np.sum((labels[rid] == j) * 1) 
            
            ltmp = ltmp / NL
            rtmp = rtmp / NR

            EL = -np.sum(ltmp * math.log2(ltmp)) 
            ER = -np.sum(rtmp * math.log2(rtmp)) 

            info_gain[i] = -1/N * (NL * EL + NR * ER)

        return candidates[np.where(info_gain == np.max(info_gain))[0]]


    # for each subsampled pts, randomly pick v and t
    def computeFeature(self,ii_ims,training_samples):
        """randomly pick a texture layout filter and compute the feature response for all 
           the ii_ims at sampled pts
        
        Arguments:
            ii_ims {list of np.array} -- list of int images(size height * width * depth)
            training_samples {list of np.array} -- list of tuple (im_id,x(vertical),y(horizontal))
        
        Return:
            res {list of int} -- list of feature response at sampled pts
            node {Node class object} node [v,t]
            
        """
        num_textons = ii_ims.shape[2]
        
        # determine v = [h,w,offset], this specifies v
        w = np.random.uniform(self.min_recSize,self.max_recSIze)
        h = np.random.uniform(self.min_recSize,self.max_recSIze)
        offset = np.random.uniform(-self.max_recSize/2,self.max_recSize/2)
        t = random.randint(0,num_textons)
        node = Node(h,w,offset,t)
        res = []

        # iterate through all subsampled pts
        for l in range(len(training_samples)):
            im_id, cur_x, cur_y = training_samples[l][0], training_samples[l][1],training_samples[l][2]
            # evaluate the feature response
            val = Node.value_at_pixel(ii_ims[im_id],cur_x,cur_y)
            res.append(val)

        return res,node

    # training
    def computeDepth(self,ii_ims,gt_ims,depth):
        """writing a recursive tree 
        
        Arguments:
            ii_ims {[type]} -- [description]
            gt_ims {[type]} -- [description]
            depth {[type]} -- [description]
        """
        
        # ending condition
        if depth == self.maxDepth: 

        


    # testing
    def evaluate(self,test_ints,test_gts):
        pass


class random_forest(object):
    def __init__(self):
        pass

