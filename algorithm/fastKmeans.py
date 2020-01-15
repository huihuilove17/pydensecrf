import numpy as np
import random
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from numba import jit
from tqdm import tqdm


from sklearn.datasets import make_blobs

random.seed(2)

def random_pick(ntotal,ncluster):
    '''
    randomly pick n starting points
    parameters: 
        ntotal: total number of points
        ncluster: number of cluster centers
    output:
        return a list of indices
    '''
    return random.sample([i for i in range(ntotal)],ncluster)


def calDist2(data,center):
    """vectorized implementation to calculate distance between datas and centers
    
    Arguments:
        data {list of np.array of shape (1,N)}-- [list of M data]
        center {list of np.array of shape (1,N)} -- [center can be only one or many]
    Return :
        np.array of distance value 
    """
    M = len(data)
    dataArr = np.array(data)
    # only one center (dist_cx_c or dist_x_cx)
    if len(center) == 1:
        centerp = center[0]
        if len(centerp.shape) == 1:
            centerp = centerp.reshape(1,len(centerp))
        centerArr = np.repeat(centerp,M,axis=0)
        
    elif len(center) == len(data):
        centerArr = np.array(center)
        
    diff = dataArr - centerArr
    return np.sum(diff**2,axis=-1)**(1./2)



def calCentersDist(centers):
    """maintain a hash table for pairs of points in points
    
    Arguments:
        points {list of vectos} -- [list of points]
        disMat {np.ndarray} -- distance matrix between points
    
    Return:
        distance matrix
    """
    K = len(centers)
    distMat = 9999999 * np.eye(K)
    for i in range(K):
        distMat[i,i+1:] = calDist2(centers[i+1:],[centers[i]])
    distMat += np.transpose(distMat) 
    return distMat



def cluster_plot(X,y):
        """plot result for clustering algorithm (assume three classes)
        
        Arguments:
            X {list of ndarr} -- [list of training data]
            y {list of int} -- [list of labels]
        """
        
        point_assignMap = np.array(y)
        training_data = np.array(X)
        class0 = training_data[np.where(point_assignMap == 0)[0]]
        class0_x = np.array([ele[0] for ele in class0])
        class0_y = np.array([ele[1] for ele in class0])

        class1 = training_data[point_assignMap == 1]
        class1_x = np.array([ele[0] for ele in class1])
        class1_y = np.array([ele[1] for ele in class1])
        
        class2 = training_data[point_assignMap == 2]
        class2_x = np.array([ele[0] for ele in class2])
        class2_y = np.array([ele[1] for ele in class2])
        
        plt.scatter(class0_x,class0_y,
                    s = 50, 
                    c = 'lightgreen',
                    marker='s', 
                    edgecolor='black',
                    label='cluster 1')

        plt.scatter(class1_x,class1_y,
            s = 50, 
            c = 'orange',
            marker='o', 
            edgecolor='black',
            label='cluster 2')


        plt.scatter(class2_x,class2_y,
            s = 50, 
            c = 'lightblue',
            marker='v', 
            edgecolor='black',
            label='cluster 3')

        plt.grid()
        




class fastKmeans2(object):
    def __init__(self):
        """initialize
        
        Arguments:
            object {none} -- [description]
            points {list of vectors} -- [training data]
            K {int} -- [number of clusser]
            eps {float} -- [description]
            max_iter {int} -- [description]
        """
        super().__init__()

        
        
    def train(self, points, K, max_iter):
        """training
        
        Arguments:
            eps {[type]} -- [description]
            max_iter {[type]} -- [description]
        """

        #initialize local variables
        training_data = np.array(points)  
        ntotals = len(points)      
        nclutsers = K
        unchanged = ntotals # total number of unchanged points
        old_point_assignMap = -1 * np.ones(ntotals)
        round = 0

        #===================================================================
        # initialize K initial clusters(later change to further use method)
        init_clusters_index = random_pick(ntotals,nclutsers)
        clusters = training_data[init_clusters_index]
        
        lowerBound = np.zeros([ntotals,nclutsers])
        recomputed = np.ones(ntotals)
        mob = np.zeros(nclutsers)
        point_assignMap = np.zeros(ntotals)
        
        t1 = time.time()
        while unchanged > 0 or round < max_iter:

            print('starting training round %i\n'%round)
            #==================================================================================================
            # first round of E step and M step
            # assign points to the nearest initalized cluster center
                
            if round == 0:
                for c in range(nclutsers):
                    # calculate dist from all points to c
                    lowerBound[:,c] = calDist2(training_data,[clusters[c]])
                
                upperBound = np.min(lowerBound,axis = 1)

                # find the min index for each row
                for i, row in enumerate(lowerBound):
                    point_assignMap[i] = np.where(row == np.min(row))[0][0]
                
                point_assignMap = np.array(list(map(int,point_assignMap))) # here, point_assignMap is the old version

                
        
            #=================================================================== 
            # repeat until convergence (E and M step)

            else:
                # calculate distance between current cluster centers
                centerDists = calCentersDist(clusters)
                mat_cx_c = np.array([centerDists[point_assignMap[i]] for i in range(ntotals)])
                s = (1/2) * np.min(mat_cx_c,axis=1)
                
                remain_index = np.where(upperBound > s)[0]
                
                # step 3
                # iterate through clusters
                for c in range(nclutsers):
                    idx1 = remain_index[np.where(point_assignMap[remain_index] != c)[0]] #absolute index: c != c(x) for unqualified point
                    idx2 = np.where(upperBound[idx1] > 1/2 * mat_cx_c[idx1,c])[0] # relative index in idx1: u(x) > 1/2 *d(c(x),c)
                    if len(idx2) == 0: continue
                    idx3 = np.where(upperBound[idx1[idx2]] > lowerBound[idx1[idx2],c])[0]  # relative index in idx2: u(x) > l(x,c) for unqualified points
                    # 3(a) - 3(b)
                    if len(idx3) > 0:
                        # 3(a) compute d(x,c(x))
                        idx3 = idx1[idx2[idx3]] #return to absolute index
                        redo = idx3[np.where(recomputed[idx3] == 1)[0]] #absolute index
                        redo_class = np.unique(point_assignMap[redo])
                        for cc in redo_class:
                            redo_pts = redo[np.where(point_assignMap[redo] == cc)[0]]
                            dist = calDist2(training_data[redo_pts],[clusters[cc]]) # d(x,c(x))
                            lowerBound[redo_pts,cc] = dist
                            upperBound[redo_pts] = dist

                        recomputed[redo] = 0

                    idx3 = idx3[np.where(upperBound[idx3] > 1/2 * mat_cx_c[idx3,c])[0]]

                    if len(idx3) == 0: continue
                    else:
                        # 3(b) compute d(x,c)
                        idx4 = idx3[np.where(upperBound[idx3] > lowerBound[idx3,c])[0]] # d(x,c_x) > l(x,c)
                        if len(idx4) == 0: continue
                        cdist = calDist2(training_data[idx4],[clusters[c]]) # d(x,c)
                        lowerBound[idx4,c] = cdist

                        tmp = np.where(cdist < upperBound[idx4])[0] 
                        idx5 = idx4[tmp] # d(x,c) < d(x,c(x))
                        point_assignMap[idx5] = c
                        upperBound[idx5] = cdist[tmp]



            #=================================================================== 
            # step 4 - 7 (for iteration 0 also)
            # find the difference btw before and after point assignment map
            diff = np.where(point_assignMap != old_point_assignMap)[0]
            diff_class = np.unique(np.array((point_assignMap[diff],old_point_assignMap[diff])).reshape(-1))
            
            old_clusters = clusters.copy()

            # update the clusters
            
            for c in diff_class:
                if c == -1: continue
                else:
                    c = int(c)
                    plus_inx = np.where(point_assignMap[diff] == c)[0]
                    minus_inx = np.where(old_point_assignMap[diff] == c)[0]
                    oldmob = mob[c]
                    mob[c] = mob[c] + len(plus_inx) - len(minus_inx)

                    # update the centers
                    
                    if mob[c] > 0:
                        clusters[c] = (clusters[c] * oldmob + np.sum(training_data[diff[plus_inx]],axis=0) - np.sum(training_data[diff[minus_inx]],axis=0))/mob[c]
                    else:
                        continue

                
                    # update lowerbound and upperbound if necessary
                    dist_c_mc = calDist2(clusters[c],old_clusters[c])
                    lowerBound[:,c] = np.maximum(lowerBound[:,c] - dist_c_mc,0)
                    
                    track = np.where(point_assignMap == c)[0]
                    upperBound[track] = upperBound[track] + dist_c_mc
            
        
            # end else
            # check stopping criterior
            unchanged = len(diff)
            round += 1
            recomputed = np.zeros(ntotals)
            old_point_assignMap = point_assignMap.copy()

            
        # end while 
        t2 = time.time()
        print('total training time %s'%(t2-t1))

        return point_assignMap


    def evaluate(self,testData):
        """evaluate test data
        
        Arguments:
            testData {list of vectors} -- [test data]
        """
        pass













if __name__ == '__main__':
    """testing
    """

    X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
    X = [np.array(ele) for ele in X]
    fastkmeans = fastKmeans2()
    ym = fastkmeans.train(X,3,1000)

    print('kmeans using sklearn')
    t1 = time.time()
    km = KMeans(
        n_clusters=3, init='random',
        n_init=10, max_iter=1000, 
        tol=1e-04, random_state=0
        )
    y_km = km.fit_predict(X)
    t2 = time.time()
    print('total training time %s'%(t2-t1))

    f1 = plt.figure(1)
    cluster_plot(X,ym)

    f2 = plt.figure(2)
    cluster_plot(X,y_km)

    plt.show(block=False)
    plt.pause(10)
    plt.close('all')