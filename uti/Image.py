'''
convert rgb image to lab image
'''
import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab
from matplotlib import pyplot as plt


def init_msrc():
    """construct a color id map for ground truth images
    """
    color_to_id = dict()
    color_to_id[np.array([128,0,0]).tobytes()] = 0
    color_to_id[np.array([0,128,0]).tobytes()] = 1
    color_to_id[np.array([128,128,0]).tobytes()] = 2 
    color_to_id[np.array([0,0,128]).tobytes()] = 3
    color_to_id[np.array([0,128,128]).tobytes()] = 4
    color_to_id[np.array([128,128,128]).tobytes()] = 5
    color_to_id[np.array([192,0,0]).tobytes()] = 6
    color_to_id[np.array([64,128,0]).tobytes()] = 7
    color_to_id[np.array([192,128,0]).tobytes()] = 8
    color_to_id[np.array([64,0,128]).tobytes()] = 9
    color_to_id[np.array([192,0,128]).tobytes()] = 10
    color_to_id[np.array([64,128,128]).tobytes()] = 11
    color_to_id[np.array([192,128,128]).tobytes()] = 12
    color_to_id[np.array([0,64,0]).tobytes()] = 13
    color_to_id[np.array([128,64,0]).tobytes()] = 14
    color_to_id[np.array([0,192,0]).tobytes()] = 15
    color_to_id[np.array([128,64,128]).tobytes()] = 16
    color_to_id[np.array([0,192,128]).tobytes()] = 17
    color_to_id[np.array([128,192,128]).tobytes()] = 18
    color_to_id[np.array([64,64,0]).tobytes()] = 19
    color_to_id[np.array([192,64,0]).tobytes()] = 20
    color_to_id[np.array([0,0,0]).tobytes()] = -1
    color_to_id[np.array([64,0,0]).tobytes()] = -2
    color_to_id[np.array([128,0,128]).tobytes()] = -3

    return color_to_id




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
    
    def get(self,op_name):
        """return different dimension for an image 
        
        Arguments:
            op_name {str} -- 'h','d','w','data'
        """
        if op_name == 'height':
            return self.height_
        elif op_name == 'width':
            return self.width_
        elif op_name == 'depth':
            return self.depth_
        elif op_name == 'data':
            return self.data_
        else:
            return self.name






def loadImage(names):
    """loading images
    
    Arguments:
        names {list of str} -- list of image names
    
    Returns:
        [type] -- [description]
    """
    res = []
    for ele in names:
        image_path = os.path.join('/home/hanhui/Documents/pydensecrf/data/msrc/Images',ele)
        im = imread(image_path)
        res.append(Image(name,im))

    return res

def rgb2labs(ims):
    """convert rgb image to lab images
    
    Arguments:
        ims {list of class object Image} -- [description]
    
    Returns:
        [list of class object Image] -- [description]
    """
    res = []
    for im in ims:
        lab_im = rgb2lab(im.get('data'))
        res.append(Image(im.get('name'),lab_im))

    return res


def loadLabelImage(names):
    """loading ground truth image 
    
    Arguments:
        names {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    gts = []
    color_map = init_msrc()
    for name in names:

        str = name.split('.')[0]+'_GT.bmp'
        image_path = os.path.join('/home/hanhui/Documents/pydensecrf/data/msrc/GroundTruth',str)
        im = imread(image_path)
        height, width, _ = im.shape
        gt_im = np.zeros((height,width))
        for j in height:
            for i in width:
                val = color_map[im[j,i,:].to_bytes()]
                gt_im[j,i] = val
        gts.append(Image(name,gt_im))

    return gts
    





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


