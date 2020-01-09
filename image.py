'''
convert rgb image to lab image
'''
from skimage.io import imread
from skimage.io import imshow
from skimage.color import rgb2lab
from feature.texton import *
import time
from matplotlib import pyplot as plt

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



