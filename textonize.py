'''
Textonize all the images 
'''
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.io import imshow
from skimage.color import rgb2lab
from feature.filterbank import FilterBank
import os
from sklearn.cluster import KMeans
from uti.loadimages import loadimages
from tqdm import tqdm
import random


#===============================================================================

def texton_visualize(lab_image,KMeans):
    nclusters = KMeans.cluster_centers_.shape[0]
    fig,axs = plt.subplots(nrows=1,ncols=2)
    axs[0].imshow(lab_image[0])
    axs[0].set_title('original image')

    axs[1].imshow(KMeans.labels_.reshape(213,320)*(255/9))
    axs[1].set_title('texton map')

    plt.show()

#===============================================================================
class Texton(object):
    """class object for texton
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self,feature):
        """initialization 
        
        Arguments:
            feature {feature class object} -- class object
        """
        self.feature_ = feature
        self.ntrain = 0  # number of training pixels
        self.ntest = 0 
        self.mean_ = None  # store the mean for the training sample data(portion of training data)
        self.transformation_ = None # store the covariance for the training sample data(portion of training data)
        self.trainTID = None
        self.testTID = None

        self.train_allFeatures = [] # list of feature response(17d vector) for 
        self.test_allFeatures = []
        self.kmeans = None

        self.feature = None

    def computeFeature(self,images): 
        """ compute mean and variance for feature response of training images
        
        Arguments:
            images {list of np.array} -- list of input images
        
        Keyword Arguments:
            samples_per_image {int} -- [description] (default: {200})
        
        Returns:
            [type] -- [description]
        """
        
        D = self.feature.getSize()
        round = 0
        all_features = []
        sample_features_id = []
        mean = np.zeros(D)
        covariance = np.zeros((D,D))
        cnt = 0

        
        # itearte through training images
        for round in tqdm(range(len(images)):
            print('processing training image %i'%round)
            feature_response = feature.evaluate_an_image(images[round]) # feature_size x height x width
            _, height, width = feature_response.shape

            # iteratively computes mean and covariance
            for j in range(height):
                for i in range(width):
                    x = feature_response[:,j,i]
                    cnt += 1
                    delta = x - mean
                    mean += delta/cnt
                    covariance += delta.reshape((len(x),1)) * (x-mean)
                    all_features.append(x)


        covariance = covariance/cnt
        U, Lambda, _ = np.linalg.svd(covariance)
        self.mean = mean
        self.transformation = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)),U.T)

        return all_features


    def fit_(self,names,feature,nTextons,samples_per_image,sample= True):


        height, width, _ = ims[0].shape
        # compute mean and variance
        all_features = self.computeTrain(ims)

        # select portion of training data
        sample_training = random.sample(range(0, len(all_features)), len(names)*samples_per_image)

        remains = [ele for ele in range(len(all_features)) if ele not in sample_training]
                
        # whitening
        X_mean = np.array(all_Features) - self.mean
=======


        X_white = np.dot(X_mean,self.transformation.T)

        # clustering using sample X
        kmeans = KMeans(n_clusters=nTextons,random_state=0,algorithm='elkan').fit(X_mean[sample_training,:])

        # evaluate remaining training pixels
        remainX_TID = kmeans.predict(X_mean[remains,:])
        sampleX_TID = kmeans.labels_

        # combine 

        lis = list(zip(sample_training,sampleX_TID)) + list(zip(remains,remainX_TID))
        lis.sort(key=lambda x: x[0])
        trainTID = [ele[1] for ele in lis]

        self.trainTID = np.array(trainTID).reshape((ntrain,height,width))
        self.kmeans = kmeans

    def evaluate(self,names):
        """compute textons for testing images
        
        Arguments:
            testing_names{list of str} -- [list of names for testing images]
        """
        ntest = len(names)
        ims = loadimages(names)
        test_all_features = []
        height, width, _ = ims[0].shape
        # iterate through images
        for round in tqdm(range(len(ims))):
            print('processing test image %i'%round)
            im = ims[round]
            feature_response = self.feature_.evaluate_an_image(im)            

            for j in range(height):
                for i in range(width):
                    x = feature_response[:,j,i]
                    test_all_features.append(x)


        test_all_features = np.array(test_all_features)
        test_all_features_white = np.dot(test_all_features-self.mean,self.transformation.T)

        test_TID = self.kmeans.predict(test_all_features_white)

        self.testTID = test_TID.reshape(ntest,height,width)
 

    def saveTextons(self,names,saving_path,mode='train'):
                          
        """should save each pixel as textondata
        
        Arguments:
            saving_path {[type]} -- [description]
        """
                          
        # saving training images
        if mode == 'train':
            for i in range(len(names)):
                dat = self.testTID[i]
                save_name = os.path.join(os.getcwd(),'texton/train/') + names[i] + '.npy' 
                np.save(save_name,dat)
        
        if mode == 'test':
            for i in range(len(names)):
                dat = self.testTID[i]
                save_name = os.path.join(os.getcwd(),'texton/test/') + names[i] + '.npy' 
                np.save(save_name,dat) 

        if mode == 'valid':
            for i in range(len(names)):
                dat = self.testTID[i]
                save_name = os.path.join(os.getcwd(),'texton/valid/') + names[i] + '.npy' 
                np.save(save_name,dat)




    def visualTextons(self):
        pass






if __name__ == '__main__':

    train_path = '/home/hanhui/Documents/pydensecrf/data/Train.txt'
    test_path  = '/home/hanhui/Documents/pydensecrf/data/Test.txt'
    train_names = []
    test_names = []

    with open(train_path,'r') as fi:
        image_files = fi.readlines()

    train_names = [ele.strip('\n') for ele in image_files][:3]

    with open(test_path,'r') as fi:
        image_files = fi.readlines()

    test_names = [ele.strip('\n') for ele in image_files][:3]

    texton = Texton()
    feature = FilterBank(5)
    nTextons = 400
    texton.fit_train(train_names,feature,nTextons)
    texton.evaluate(test_names)
    texton.saveTextons('/home/hanhui/Documents/pydensecrf/texton')





