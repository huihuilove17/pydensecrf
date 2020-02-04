'''
py file for texton class object
'''
import numpy as np
import random
from uti.Image import loadImages
from tqdm import tqdm
from sklearn.cluster import KMeans
import os
import h5py

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
        self.kmeans = None
        self.trainTID = None
        self.testTID = None
        self.validTID = None
        self.trainNames = None
        self.testNames = None
        self.validNames = None


    def fit_(self,images): 
        """ compute mean and variance for feature response of training images
        
        Arguments:
            images {list of np.array} -- list of input images (we use portion of training images to train)
        
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


    def train_kmeans(self,names,nTextons,samples_per_image,ifLAB=False):

        print('start training kmeans')
        # loading training images
        ims = loadImages(names,ifLAB)
        height, width, _ = ims[0].shape
        ntrain = len(names)

        # compute mean and variance
        all_features = self.fit_(ims) # list of feature response for each image, feature response of size (height,width,feature_size)

        # sampling
        sample_training = []
        for ele in all_features:
            height, width, _ = ele.shape
            row_idx = np.random.choice(height,samples_per_image)
            col_idx = np.random.choice(width,samples_per_image)
            for j in range(samples_per_image):
                sample_training.append(ele[row_idx[j],col_idx[j],:])

        # whitening
        X_mean = np.array(sample_training) - self.mean
        X_white = np.dot(X_mean,self.transformation.T)
        self.kmeans = KMeans(n_clusters=nTextons,random_state=0,algorithm='elkan').fit(X_white)
        
        print('finishing training kmeans')


    def evaluate(self,names,mode='train'):
        """compute textons for testing images
        
        Arguments:
            names{list of str} -- [list of names for images]
            mode -- specify whether it is test image or validate images
        """
        # loading test images
        print('now evaluate %s images!'%mode)
        ims = loadImages(names)
        nims = len(ims)
        all_features = []

        # iterate through images
        for round in tqdm(range(nims)):
            im = ims[round]
            feature_response = self.feature_.evaluate_an_image(im)            
            height, width, _ = im.shape

            for j in range(height):
                for i in range(width):
                    x = feature_response[j,i,:]
                    all_features.append(x)

        # whitening
        all_features = np.array(all_features)
        all_features_white = np.dot(all_features-self.mean,self.transformation.T)

        imageTID = self.kmeans.predict(all_features_white)
        imageTID_final = []
        num1 = num2 = 0
        # account for different size of image
        for l in range(nims):
            height,width,_ = ims[l].shape
            num2 += height*width
            imageTID_final.append(np.array(imageTID[num1:num2]).reshape(height,width))
            num1 = num2
       
        if mode == 'train':
            self.trainTID = imageTID_final
            self.trainNames = names
        elif mode == 'test':
            self.testTID = imageTID_final
            self.testNames = names
        else:
            self.validTID = imageTID_final
            self.validNames = names

    
    def save_by_image(self,saving_path,mode='train'):
        """save textons by images
        
        Arguments:
            saving_path {[type]} -- [description]
        
        Keyword Arguments:
            mode {str} -- [description] (default: {'train'})
        """
        pass


    def saveTextons(self,saving_path,mode = 'train'):
                          
        """should save each pixel as textondata
        
        Arguments:
            saving_path {[type]} -- [description]
        """
        feat_name = self.feature_.get_name() # get feature name
        print('saving textons!')
        if mode == 'train':
            save_name = os.path.join(os.getcwd(),'data/texton/') + 'train_msrc_' + feat_name + '.h5' 
            f = h5py.File(save_name,'w')
            grp = f.create_group('list of images')
            dat = self.trainTID
            names = self.trainNames
            for i in tqdm(range(len(names))):
                grp.create_dataset(names[i],data=dat[i],compression='gzip',compression_opts=9)
            f.close()

        elif mode == 'test':
            save_name = os.path.join(os.getcwd(),'data/texton/') + 'test_msrc_' + feat_name + '.h5' 
            f = h5py.File(save_name,'w')
            grp = f.create_group('list of images')
            dat = self.testTID
            names = self.testNames
            for i in tqdm(range(len(names))):
                grp.create_dataset(names[i],data=dat[i],compression='gzip',compression_opts=9)
            f.close()

        else:
            save_name = os.path.join(os.getcwd(),'data/texton/') + 'valid_msrc_' + feat_name + '.h5' 
            f = h5py.File(save_name,'w')
            grp = f.create_group('list of images')
            dat = self.validTID
            names = self.validNames
            for i in tqdm(range(len(names))):
                grp.create_dataset(names[i],data=dat[i],compression='gzip',compression_opts=9)
            f.close()
    

if __name__ == '__main__':
    pass