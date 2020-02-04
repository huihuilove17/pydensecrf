'''
config file
'''

config = {

    # color encoding
    'id_to_color': {
        0: [128,0,0],
        1: [0,128,0],
        2: [128,128,0],
        3: [0,0,128],
        4: [0,128,128],
        5: [128,128,128],
        6: [192,0,0],
        7: [64,128,0],
        8: [192,128,0],
        9: [64,0,128],
        10: [192,0,128],
        11: [64,128,128],
        12: [192,128,128],
        13: [0,64,0],
        14: [128,64,0],
        15: [0,192,0],
        16: [128,64,128],
        17: [0,192,128],
        18: [128,192,128],
        19: [64,64,0],
        20: [192,64,0],
        -1: [0,0,0],
        -2: [64,0,0],
        -3: [128,0,128]
    },

    'color_to_id': {
        (128,0,0): 0,
        (0,128,0): 1,
        (128,128,0):2,
        (0,0,128): 3,
        (0,128,128):4,
        (128,128,128):5,
        (192,0,0):6,
        (64,128,0):7,
        (192,128,0):8,
        (64,0,128):9,
        (192,0,128):10,
        (64,128,128):11,
        (192,128,128):12,
        (0,64,0):13,
        (128,64,0):14,
        (0,192,0):15,
        (128,64,128):16,
        (0,192,128):17,
        (128,192,128):18,
        (64,64,0):19,
        (192,64,0):20,
        (0,0,0):-1,
        (64,0,0):-2,
        (128,0,128):-3
    },

    # image names path 
    'train_path': '/Users/huihuibullet/Documents/project/pydensecrf-1/data/Train.txt',
    'test_path': '/Users/huihuibullet/Documents/project/pydensecrf-1/data/Test.txt',
    'valid_path': '/Users/huihuibullet/Documents/project/pydensecrf-1/data/Validation.txt',
    'texton_path': '/Users/huihuibullet/Documents/project/pydensecrf-1/data/texton',
    
    # texton parameters
    'kappa': 5,
    'filter_nTextons': 400,
    'color_nTextons': 128,
    'location_nTextons': 144,
    'hog_nTextons': 150, 
    'level': 0,
    'samples_per_image': 200,
    
    # compute integral images
    'texton_type': ['filterbank','color'],
    'subsample': 1,
    'intIms_path':'/Users/huihuibullet/Documents/project/pydensecrf-1/data/ints'
    
}

if __name__ == '__main__':
    pass

