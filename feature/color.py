'''
implement color feature
'''

class Color(object):
    def __init__(self):
        # use cielab color image space
        self.size_ = 3
        self.name_ = 'color'

    def evaluate_an_image(self,lab_image):
        return lab_image

    def get_name(self):
        return self.name_
    
    def get_size(self):
        return self.size_