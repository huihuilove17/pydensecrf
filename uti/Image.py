'''
convert rgb image to lab image
'''
from skimage.io import imread
from skimage.io import imshow
from skimage.color import rgb2lab
from feature.texton import *
import time
from matplotlib import pyplot as plt




class Image(object):
    """class object for image
    
    Arguments:
        object {[type]} -- [description]
    """
    def __init__(self,name,im):
        self.name_ = name
        self.height_ = im.shape[0]
        self.width_ = im.shape[1]
        self.depth_ = im.shape[2] 
        self.data_ = im
    
    def get(op_name):
        """return different dimension for an image 
        
        Arguments:
            op_name {str} -- 'h','d','w','data'
        """
        if op_name == 'h':
            return self.height_
        elif op_name == 'w':
            return self.width_
        elif op_name == 'd':
            return self.depth_
        else:
            return self.data_


class loadImage(object):



def loadImage(names):
    ims = []
    lab_ims = []
    gt_ims = []
    id_ims = [] 
    colorID = init_msrc()
    idColor = id_to_color()
    for ele in file_names:
        image_path = os.path.join('/home/hanhui/Documents/pydensecrf/data/msrc/Images',ele)
        #gt_path = os.path.join('/home/hanhui/Documents/pydensecrf/data/msrc/GroundTruth',ele)
        im = imread(image_path)
        lab_im = rgb2lab(im)
        #gt_im = imread(gt_path)
        #id_im = np.zeros((im.shape[0],im.shape[1])) 
        # map gt_im's pixels to id
        # convert to id image
        '''
        for j in range(gt_im.shape[0]):
            for i in range(gt_im.shape[1]):
                tmp = gt_im[j,i,:]
                if tmp in colorID:
                    id_im[j,i] = colorID[tmp]
                else:
                    id_im[j,i] = 0
        '''

        ims.append(im)
        lab_ims.append(lab_im)
        #gt_ims.append(gt_im)
        #id_ims.append(id_im)

    #return ims, lab_ims, gt_ims, id_ims, idColor,file_names
    return ims,lab_ims











def loadLabelImage():
    pass













if __name__ == "__main__":
    file = '~/Documents/course_in_cu/Postgraduate Course/image_segmentation/pydensecrf/data/msrc/8_30_s.bmp'
    image = imread(file)
    image1 = rgb2lab(image)

    t1 = time.time()
    conv_filter = FilterBank(400)
    res = conv_filter.evaluate_an_image(image1)
    t2 = time.time()

    print("elapsed time for processing an image using filter bank: %s"%(t2-t1))
    for i in range(17):
        print("showing image %i"%i)
        imshow(res[:,:,i]*255)
        plt.show()



