'''
Textonize all the images 
'''
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.io import imshow
from skimage.color import rgb2lab
from feature.filterbank import FilterBank
import os
from sklearn.cluster import KMeans
from uti.loadimages import loadimages

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
    def __init__(self):
        self.ntrain = 0  # number of training pixels
        self.ntest = 0 
        self.mean = None  # store the mean for the training sample data(portion of training data)
        self.transformation = None # store the covariance for the training sample data(portion of training data)
        self.trainTID = None
        self.testTID = None
        self.sampleFeaturesIDX = [] # list of feature response(17d vector) id for portion of training data
        self.train_allFeatures = [] # list of feature response(17d vector) for 
        self.test_allFeatures = []
        self.Kmeans = None
        self.samples_per_image = None
        self.nclass = None 
        self.nTextons = None
        self.train_filenames= None
        self.test_filenames = None
        self.feature = None

    def computeTrain(self,images,feature,nTextons,samples_per_image = 200): 
        """compute texton for training images
        
        Arguments:
            images {list of np.array} -- list of input images
            feature {feature class} -- 
            nTextons {int} -- number of texton
        
        Keyword Arguments:
            samples_per_image {int} -- number of sample pixels (default: {200})
            if_sample {bool} -- [description] (default: {True})
            if_train {bool} -- [description] (default: {True})
        """
        self.feature = feature
        self.featureSize = feature.getSize()
        self.samples_per_image = samples_per_image
        round = 0
        D = feature.getSize()
        all_features = []
        sampleFeaturesIDX = []
        # training
        cnt = 0
        mean = np.zeros(D)
        covariance = np.zeros((D,D))
        # itearte through training images
        for image in images:
            print('processing training image %i'%round)
            feature_response = feature.evaluate_an_image(image) #compute feature response at full resolution
            # iterate at full resolution
            height = feature_response.shape[1] # input image's rows
            width = feature_response.shape[2]  # inut image's cols
            npixels = round * height * width
            for j in range(height):
                for i in range(width):
                    x = feature_response[:,j,i]
                    cnt += 1
                    delta = x - mean
                    mean += delta/cnt
                    covariance += delta.reshape((len(x),1)) * (x-mean)
                    all_features.append(x)
            # random pick 
            num = 0
            while num < samples_per_image:
                x_rand = randint(0,width-1)
                y_rand = randint(0,height-1)
                idx = npixels + y_rand*width + x_rand
                if idx not in sampleFeaturesIDX: 
                    sampleFeaturesIDX.append(idx) # absolute index in all features
                    num += 1
                else:
                    continue

            round += 1

        covariance = covariance/cnt
        U, Lambda, _ = np.linalg.svd(covariance)
        self.sampleFeaturesIDX = sampleFeaturesIDX
        self.transformation = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)),U.T)
        self.mean = mean
        self.height = height # assume all the images have the same shape
        self.width = width
        self.train_allFeatures = all_features

    def fit_train(self,training_names,feature,nTextons,sample= True):
        """loading training pixels. Perform sampling and whitening
            use portion of training pixels to train, but save all the feature responses in the training images
        
        Arguments:
            training_names {list of strings}-- [list of filenames of trainging images]
            training_label_path {txt file}-- [storing the true labels for training images]
            feature {class object} -- [compute the feature response]
        """
        # loading training images and the true labels
        self.ntrain = len(training_names)
        self.train_filenames = training_names

        ims = loadimages(training_names)
        train_ims = ims[1]

        self.computeTrain(train_ims,feature,nTextons,samples_per_image = 200)
        samples_per_image = self.samples_per_image

        remainX_id = [ele for ele in range(len(self.train_allFeatures)) if ele not in self.sampleFeaturesIDX]
                
        # whitening
        X_mean = np.array(self.train_allFeatures) - self.mean
        X_white = np.dot(X_mean,self.transformation.T)

        # clustering using sample X
        kmeans = KMeans(n_clusters=nTextons,random_state=0,algorithm='elkan').fit(X_mean[self.sampleFeaturesIDX,:])

        # evaluate remaining training pixels
        remainX_TID = kmeans.predict(X_mean[remainX_id,:])
        sampleX_TID = kmeans.labels_

        # combine 
        lis = list(zip(self.sampleFeaturesIDX,sampleX_TID)) + list(zip(remainX_id,remainX_TID))
        lis.sort(key=lambda x: x[0])
        trainTID = [ele[1] for ele in lis]

        self.trainTID = np.array(trainTID).reshape((self.ntrain,self.height,self.width))
        self.train_filenames = training_names
        self.Kmeans = kmeans

    def evaluate(self,testing_names):
        """compute textons for testing images
        
        Arguments:
            testing_names{list of str} -- [list of names for testing images]
        """
        self.test_filenames = testing_names
        self.ntest = len(testing_names)
        
        ims = loadimages(testing_names)
        test_ims = ims[1]
        test_all_features = []

        # iterate through images
        for round,im in enumerate(test_ims):
            print('processing test image %i'%round)
            feature_response = self.feature.evaluate_an_image(im) #compute feature response at full resolution
            # iterate at full resolution
            height = feature_response.shape[1] # input image's rows
            width = feature_response.shape[2]  # inut image's cols
            
            for j in range(height):
                for i in range(width):
                    x = feature_response[:,j,i]
                    test_all_features.append(x)

        all_features = np.array(test_all_features)
        all_features_white = np.dot(all_features-self.mean,self.transformation.T)

        test_TID = self.Kmeans.predict(all_features_white)

        self.testTID = test_TID.reshape(self.ntest,height,width)
 

    def saveTextons(self,saving_path):
        """should save each pixel as textondata
        
        Arguments:
            saving_path {[type]} -- [description]
        """
        # saving train textons     
        for i, name in enumerate(self.train_filenames):
            data = self.trainTID
            save_name = os.path.join(os.getcwd(),'texton/train/') + name + '.npy' 
            np.save(save_name,data)
    
        for i, name in enumerate(self.test_filenames):
            data = self.testTID
            save_name = os.path.join(os.getcwd(),'texton/test/') + name + '.npy' 
            np.save(save_name,data)



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





