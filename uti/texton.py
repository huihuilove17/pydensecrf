'''
py file for texton class object
'''
import numpy as np
import random
from tqdm import tqdm

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

        self.trainTID = np.array(trainTID).reshape((ntrain,height,width)) #by the order in the names
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
        name = self.feature.get_name()
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


