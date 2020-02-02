'''
py file for texton class object
'''
import numpy as np
import random
from uti.Image import loadImages
from tqdm import tqdm
from sklearn.cluster import KMeans
import os

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
        self.mean_ = None  # store the mean for the training sample data(portion of training data)
        self.transformation_ = None # store the covariance for the training sample data(portion of training data)
        self.trainTID = None
        self.testTID = None
        self.kmeans = None


    def computeFeature(self,images): 
        """ compute mean and variance for feature response of training images
        
        Arguments:
            images {list of np.array} -- list of input images
        
        Keyword Arguments:
            samples_per_image {int} -- [description] (default: {200})
        
        Returns:
            [type] -- [description]
        """
        D = self.feature_.get_size()
        round = 0
        all_features = []
        mean = np.zeros(D)
        covariance = np.zeros((D,D))
        cnt = 0

        print('now compute features for training images')
        # itearte through training images
        for round in tqdm(range(len(images))):
            #print('processing training image %i'%round)
            feature_response = self.feature_.evaluate_an_image(images[round]) # height x width x feature_size
            height,width,_ = feature_response.shape

            single_feature = []
            # iteratively computes mean and covariance
            for j in range(height):
                for i in range(width):
                    x = feature_response[j,i,:]
                    cnt += 1
                    delta = x - mean
                    mean += delta/cnt
                    covariance += delta.reshape((len(x),1)) * (x-mean)
            
            all_features.append(feature_response)

        covariance = covariance/cnt
        U, Lambda, _ = np.linalg.svd(covariance)
        self.mean = mean
        self.transformation = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)),U.T)

        return all_features


    def fit_(self,names,nTextons,samples_per_image):

        print('start training kmeans')
        # loading training images
        ims = loadImages(names)
        height, width, _ = ims[0].shape
        ntrain = len(names)

        # compute mean and variance
        all_features = self.computeFeature(ims) # list of feature response for each image, feature response of size (height,width,feature_size)
        
        all_features_lis = [ele[j,i,:] for ele in all_features for j in range(ele.shape[0]) for i in range(ele.shape[1])]

        # select portion of training data
        sample_training = random.sample(range(0, len(all_features_lis)), len(names)*samples_per_image)

        remains = [ele for ele in range(len(all_features_lis)) if ele not in sample_training]
                
        # whitening
        X_mean = np.array(all_features_lis) - self.mean
        X_white = np.dot(X_mean,self.transformation.T)

        # clustering using sample X
        kmeans = KMeans(n_clusters=nTextons,random_state=0,algorithm='elkan').fit(X_white[sample_training,:])

        # evaluate remaining training pixels
        remainX_TID = kmeans.predict(X_white[remains,:])
        sampleX_TID = kmeans.labels_

        # combine 
        lis = list(zip(sample_training,sampleX_TID)) + list(zip(remains,remainX_TID))
        lis.sort(key=lambda x: x[0])
        trainTID = [ele[1] for ele in lis]

        trainTID_final = []
        num1 = num2 = 0
        for l in range(len(all_features)):
            height,width,_ = all_features[l].shape
            num2 += height*width
            trainTID_final.append(np.array(trainTID[num1:num2]).reshape(height,width))
            num1 = num2

        self.trainTID = trainTID_final #by the order in the names
        self.kmeans = kmeans

    def evaluate(self,names):
        """compute textons for testing images
        
        Arguments:
            testing_names{list of str} -- [list of names for testing images]
        """
        # loading test images
        print('now evaluate test images!')

        ntest = len(names)
        ims = loadImages(names)
        test_all_features = []

        # iterate through images
        for round in tqdm(range(len(ims))):
            im = ims[round]
            feature_response = self.feature_.evaluate_an_image(im)            
            height, width, _ = im.shape

            for j in range(height):
                for i in range(width):
                    x = feature_response[j,i,:]
                    test_all_features.append(x)


        test_all_features = np.array(test_all_features)
        test_all_features_white = np.dot(test_all_features-self.mean,self.transformation.T)

        testTID = self.kmeans.predict(test_all_features_white)

        testTID_final = []
        num1 = num2 = 0
        for l in range(ntest):
            height,width,_ = ims[l].shape
            num2 += height*width
            testTID_final.append(np.array(testTID[num1:num2]).reshape(height,width))
            num1 = num2
       
        self.testTID = testTID_final
 

    def saveTextons(self,saving_path,mode = 'train'):
                          
        """should save each pixel as textondata
        
        Arguments:
            saving_path {[type]} -- [description]
        """
        name = self.feature_.get_name() # get feature name
        # saving training images
        if mode == 'train':
            dat = self.trainTID
            save_name = os.path.join(os.getcwd(),'data/texton/train/') + 'msrc_' + name + '.npy' 
            np.save(save_name,dat)
        
        if mode == 'test':
            dat = self.testTID
            save_name = os.path.join(os.getcwd(),'data/texton/test/') + 'msrc_' + name + '.npy' 
            np.save(save_name,dat)
        

    def visualTextons(self):
        pass


if __name__ == '__main__':
    pass