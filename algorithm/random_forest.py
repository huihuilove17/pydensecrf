'''
implementation of decision tree and random forest,based on the 
implementation of semantic texton forests
'''
import numpy as np
import random 
import math
from numba import njit
from uti.computeInt import computeInt


#=====================
# class object 
#=====================

class Node(object):
    """node object in the tree. Each node is specified by relative position offset, window size and 
       texton id
    
    Arguments:
        object {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    def __init__(self,h=0,w=0,offset=0,t=0,depth=0,isLeaf=False):
        self.h_ = h
        self.w_ = w
        self.t_ = t 
        self.offset_ = offset
        self.depth_ = depth
        self.isLeaf_ = isLeaf
        self.thresh_ = None
        self.class_ = None 
        self.left = None
        self.right = None
    
    def value_at_pixel(self,ii_im,i,j):
        """evalute the feature response at position (i,j)
        
        Arguments:
            ii_im {[type]} -- [description]
            i {[type]} -- [current col index]
            j {[type]} -- [current row ind]
        
        Returns:
            [type] -- [description]
        """

        hegiht,width,_ = ii_im.shape
        t = self.t

        x2 = j + self.offset
        y2 = i + self.offset
        if x2 < 0: x2 = 0
        if y2 < 0: y2 = 0

        x1 = x2 + self.h
        y1 = y2 + self.w

        if x1 >= hegiht: x1 = hegiht-1
        if y1 >= width: y1 = width-1

        return ii_im[x1,y1,t] + ii_im[x2,y2,t] - ii_im[x1,y2,t] - ii_im[x2,y1,t]

    
    # reset 
    def reset(self,feature):
        """reset all the parameters for the node
        
        Arguments:
            feature {list} -- [h,w,offset,t]
        """
        self.h_ = feature[0]
        self.w_ = feature[1]
        self.t_ = feature[3]
        self.offset_ = feature[2] 

        
    
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
        self.numNodes = 0
        self.root = None


    def computeBestThresh(self,values,labels):
        """find the optimal threshold for the current node,split the 
            node by max info gain
        
        Arguments:
            values {[type]} -- [description]
            labels {list of int} -- follows the order in the values
            node {[type]} -- [description]
        """
        candidates = np.randn(self.numThreshold) * np.std(values) + np.mean(values)
        info_gain = np.ones(self.numThreshold) * float('-inf')# should be replace by MIN_INT
        
        # compute info gain for each threshold
        # for left and right branch, calculate shannon entropy for each class seperately
        for i in range(self.numThreshold):
            gain = 0
            lid = np.where(values < candidates[i])[0] # id of left brance
            rid = np.where(values >= candidates[i])[0] # id of right brance
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

        best_infoGain = np.max(info_gain)        

        return candidates[np.where(info_gain == best_infoGain)[0]], best_infoGain


    # for each subsampled pts, randomly pick v and t
    @njit
    def computeFeature(self,ii_ims,gt_ims,training_samples):
        """randomly pick a texture layout filter and compute the feature response for all 
           the ii_ims at sampled pts
        
        Arguments:
            ii_ims {list of np.array} -- list of int images(size height * width * depth)
            training_samples {list of np.array} -- list of tuple (im_id,x(vertical),y(horizontal))
            scaling {int} -- scaling factor, like 3 or 5
        Return:
            res {list of int} -- list of feature response at pixels(we will later sample pts before calculate)
            node {Node class object} node [v,t]
            
        """
        
        # determine v = [h,w,offset], this specifies v
        w = np.random.uniform(self.min_recSize,self.max_recSIze)
        h = np.random.uniform(self.min_recSize,self.max_recSIze)
        offset = np.random.uniform(-self.max_recSize/2,self.max_recSize/2)
        t = random.randint(0,num_textons)
        node = Node(h,w,offset,t)
        res = []
        labels = []

        for ele in training_samples:
            im_id, cur_x, cur_y = ele[0],ele[1],ele[2]
            val = node.value_at_pixel(ii_ims[im_id],cur_x,cur_y)
            res.append(val)
            labels.append(gt_ims[l],cur_x,cur_y)
        
        feature = [h,w,offset,t]

        return res,labels,feature


#==================================================================================================
    # training
    def computeDepth(self,node,ii_ims,gt_ims,depth,training_samples):
        """writing a recursive tree 
        
        Arguments:
            node {Node class object} -- current node
            ii_ims {list of np.array} -- list of ii_images of size (height,width,depth) 
            gt_ims {[type]} -- [description]
            training_samples {list of tuples} -- list of (image_id, currX, currY)
            depth {int} -- current depth for the node
        """
        
        print('processing node %i'%self.numNodes)
        
        # leaf node
        if depth >= self.maxDepth or len(np.unique(gt_ims)) == 1:
            # find the class that appear most
            labels = []
            for ele in training_samples:
                im_id, cur_x, cur_y = ele[0],ele[1],ele[2]
                labels.append(gt_ims[im_id][cur_x,cur_y])
            best_class = max(set(labels),key=labels.count)

            # set the node
            node.class_ = best_class
            node.isLeaf_ = True
            node.depth_ = depth
            self.numNodes += 1

        else:
            bestScore = float('-inf')
            bestThresh = float('-inf')
            bestFeature = []
            bestVals = []

            # finding the best [v,t,theta]
            # every node has multiple possible features
            for i = 1:self.numFeature:
                vals, labels, feature  = self.computeFeature(ii_ims,gt_ims,training_samples) # vals follows the order of training samples
                thresh, score = self.computeBestThresh(vals,labels)
                if score > bestScore:
                    bestScore = score
                    bestFeature = feature
                    bestVals = vals
                    bestThresh = thresh                
            
            # reset node parameters [v,t,theta]
            node.reset(bestFeature)
            self.numNodes += 1

            # do the splitting
            left = training_samples[np.where(np.array(bestVals) <= bestThresh)[0]]
            right = training_samples[np.where(np.array(bestVals) > bestThresh)[0]]

            node.left = self.computeDepth(Node(),ii_ims,gt_ims,depth+1,left)
            node.right= self.computeDepth(Node(),ii_ims,gt_ims,depth+1,right) 

        return node 

#==================================================================================================

    # fit a decision tree
    def fit_(self,ii_ims,gt_ims,scaling):
        """use portion data to train, use all data to fill
        
        Arguments:
            ii_ims {[type]} -- [description]
            gt_ims {[type]} -- [description]
            scaling {[type]} -- [description]
        """

        # need to implement the scaling here !!!
        #iterate through each image
        portion_train = []
        all_train = []

        for l in range(len(ii_ims)):
            height, width, _ = ii_ims[l].shape
            for j in range(height):
                for i < range(width):
                    jj = scaling * j
                    ii = scaling * i
                    all_train.append((l,i,j))
                    if jj < height and ii < width:
                        portion_train.append((l,ii,jj))
                
        # use portion of data to train (could use all data to train)
        self.root = self.computeDepth(Node(),ii_ims,gt_ims,0,portion_train)


    # testing
    @njit
    def evaluate_at_image(self,ii_im,gt_im):
        
        # calculate pixel accuracy P-ACC = 1/N * (sum_{i}^{N} \delta(yi,y_i))
        height, width, _ = ii_im.shape
        yhat = np.zeros((height,width))
        for j in range(height):
            for i in range(width):
                node = self.root
                while not node.isLeaf_:
                    val = node.value_at_pixel(ii_im,i,j)
                    if val <= node.thresh_:
                        node = node.left
                    else:
                        node = node.right
                
                yhat[j,i] = node.class_
        

        diff = len(np.where(yhat,gt_im)[0])
        return diff/(height*width), yhat
        
        
    


class random_forest(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    
