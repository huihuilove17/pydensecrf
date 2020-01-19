'''
main file for xgboost,quick try 
'''
import numpy as np
from uti.loadtextons import loadtextons
from uti.Image import loadLabelImage
import xgboost as xgb
from numba import njit


# using jit to speed up
def data_formulate(texton_path,names,texton_type):
    """reformulate the data into xgboost format
    
    Arguments:
        texton_path {str} -- path for texton data
        names {list of str} -- [description]
    """

    train_textons = loadtextons(texton_path,texton_type)    # list of images(size: height, width, depth)
    train_labels = loadLabelImage(names) #list of images(size: height,width)
    height, width, depth = train_textons[0].shape

    # reformulate the data 
    big_X = np.zeros((len(train_textons)*height*width,depth))
    big_y = np.zeros(len(train_textons)*height*width)

    # slow loops
    for l in range(len(train_textons)):
        for j in range(height):
            for i in range(width):
                cid = l*height*width + j*width + i
                big_X[cid,:] = train_textons[l][j,i,:]
                big_y[cid] = train_labels[l][j,i]
    
    #formatted_data = xgb.DMatrix(big_X, label=big_y)
    
    return big_X, big_y
    

if __name__ == '__main__':

    #===============
    # setting
    #===============
 
    texton_type = ['filterbank','color','location']
    train_texton_path = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/texton/train'
    test_texton_path = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/texton/test'
    train_files = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/Train.txt'
    test_files = '/Users/huihuibullet/Documents/project/pydensecrf-1/data/Test.txt'

    # loading training and testing names
    with open(train_files,'r') as f1:
        train_names = f1.readlines()
    
    with open(test_files,'r') as f2:
        test_names = f2.readlines()


    train_names = [ele.strip('\n') for ele in train_names]
    test_names = [ele.strip('\n') for ele in test_names]
    
    #===============
    # training
    #===============
    # setting parameters
    param = {'eta': 0.3, 'max_depth': 3,'objective': 'multi:softprob','num_class': 21} 

    train_data, train_target = data_formulate(train_texton_path,train_names,texton_type)
    D_train = xgb.DMatrix(train_data, label=train_target)

    model = xgb.train(param,D_train,20)

    #===============
    # testing
    #===============
    
    test_data, test_target= data_formulate(test_texton_path,test_names,texton_type)
    D_test= xgb.DMatrix(test_data, label=test_target)

    test_label = model.predict(D_test)

    # reformulate test data
    test_label = test_label.reshape(len(test_names),height,width)



