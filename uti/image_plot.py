'''
helper function to plot the graph
'''

import matplotlib.pyplot as plt


def plot_result(original_im,gt_im.predict_map,config_map):
    """plot 2 * 2 subgraph 
    ul: original image
    ur: residual image
    ll: gt_im
    lr: predict_map
    
    Arguments:
        original_im {[type]} -- [description]
        gt_im {[type]} -- [description]
        diff_im {[type]} -- [description]
    """

    # color encoding
    predict_im = np.zeros(original_im.shape)
    h, w, _ = predict_im.shape
    for j in range(h):
        for i in range(w):
            predict_im[j,i,:] = config_map[predict_map[j,i]]
    
    diff_im = (gt_im != predict_im)*1

    ax1 = plt.subplot(2,2,1)
    ax1.set_title('original image')
    ax1.imshow(original_im)

    ax2 = plt.subplot(2,2,2)
    ax2.set_title(' residual image')
    ax2.imshow(diff_im)

    ax3 = plt.subplot(2,2,3)
    ax3.set_title('groundtruth image')
    ax3.imshow(gt_im)

    ax4 = plt.subplot(2,2,4)
    ax4.set_title('predict image')
    ax4.imshow(predict_im) 

    plt.tight_layout()
    plt.show()

