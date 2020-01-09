'''
Textonize all the images 
'''
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.io import imshow
from skimage.color import rgb2lab
from feature import filterbank
import os
from sklearn.cluster import KMeans
from uti.loadimages import loadimages

#===============================================================================
#@profile

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
        self.transformation_ = None # store the covariance for the training sample data(portion of training data)
        self.trainTID = None
        self.testTID = None
        self.sampleFeaturesIDX = [] # list of feature response(17d vector) id for portion of training data
        self.allFeatures = [] # list of feature response(17d vector) for 
        self.Kmeans = None
        self.samples_per_image = None
        self.nclass = None 
        self.nTextons = None
        self.train_filenames= None
        self.test_filenames = None
        self.feature = None

    def computeFeature(self,images,feature,nTextons,samples_per_image = 200, if_train=True):
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
        self.ntrain = len(images)
        round = 0
        all_features = []
        sampleFeaturesIDX = []
        # training
        if if_train:
            cnt = 0
            for image in images:
                print('processing training image %i'%round)
                feature_response = filter.evaluate_an_image(image) #compute feature response at full resolution
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


        self.transformation = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)),U.T))
        self.mean = mean
        self.allFeatures = all_features
        

    def fit_train(self,training_path,feature,nTextons,sample= True):
        """loading training pixels. Perform sampling and whitening
            use portion of training pixels to train, but save all the feature responses in the training images
        
        Arguments:
            training_path {txt file}-- [storing the filenames for training images]
            training_label_path {txt file}-- [storing the true labels for training images]
            feature {class object} -- [compute the feature response]
        """
        # loading training images and the true labels
        train_ims, gt_ims, id_ims, id_to_color = loadimages(training_path)
        computeFeature(train_ims,feature,nTextons,samples_per_image = 200, if_sample=sample) 
        samples_per_image = self.samples_per_image

        remainX_id = [ele for ele in range(len(self.allFeatures)) if ele not in self.sampleFeaturesIDX]
                
        # whitening
        X_mean = np.array(self.allFeatures) - self.mean
        X_white = np.doc(X_mean,self.transformation.T)

        # clustering using sample X
        kmeans = KMeans(n_clusters=nTextons,random_state=0,algorithm='elkan').fit(X_mean[self.sampleFeaturesIDX,:])

        # evaluate remaining training pixels
        remainX_TID = kmeans.predict(X_mean[remainX_id,:])
        sampleX_TID = kmeans.labels_

        # combine 
        lis = list(zip(self.sampleFeaturesIDX,sampleX_TID)) + list(zip(remainX_id,remainX_TID))
        lis = sorted(lis,lambda x: x[0])

        self.trainTID = np.array(lis).reshape()
        self.train_filenames = training_path

    def evaluate(self,testing_path):
        
        test_ims, gt_ims, id_ims, id_to_color = loadimages(testing_path)
        all_features = []

        for i,im in enumerate(test_ims):
        
            print('processing test image %i'%i)
            feature_response = self.feature.evaluate_an_image(im) #compute feature response at full resolution
            D = self.
            # iterate at full resolution
            height = feature_response.shape[1] # input image's rows
            width = feature_response.shape[2]  # inut image's cols
            
            
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
            self.sampleFeaturesIDX = sampleFeaturesIDX


        self.transformation = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)),U.T))
        self.mean = mean
        self.allFeatures = all_features
 





    def saveTextons(self,saving_path):
        """should save each pixel as textondata
        what to save?
            
        
        Arguments:
            saving_path {[type]} -- [description]
        """     



    def visualTextons(self):
        pass



if __name__ == '__main__':

    train_path = '/home/hanhui/Documents/pydensecrf/data/Train.txt'
    with open(train_path,'r') as fi:
        image_files = fi.readlines()

    image_files = [ele.strip('\n') for ele in image_files]
    images = []
    lab_images = []
    textons = []

    for i in range(3):
        path = os.path.join('~/Documents/pydensecrf/data/msrc/Images',image_files[i])
        image = imread(path)
        lab_image = rgb2lab(image)
        images.append(image)
        lab_images.append(lab_image)

    print('start training!') 
    filter = filterbank.FilterBank(15)
    kmeans = computeFeature(lab_images,filter,4)

    #texton_visualize(lab_images,kmeans)
