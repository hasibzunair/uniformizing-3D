import os 
import matplotlib.pyplot as plt
import numpy as np


def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.
    INPUT
        directory: Folder to be created, called as "folder/".
    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
def get_number(filename):
    return int(filename[:filename.find('.')])

def sort_paths(paths):
    paths.sort(key = get_number)
    return paths

MIN_BOUND = -1000 # anything below this we are not interested in 
MAX_BOUND = 700   # anything above too

def normalize(image):
    '''
    Our values currently range from -3000 to around 2400. Anything above 400 is not interesting to us, as these are simply bones with different radiodensity. Refer to histogram plot
    '''
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
    
    
PIXEL_MEAN = 0.25

def zero_center(image):
    '''
    Zero centering: As a final preprocessing step, it is advisory to zero center your data so that your mean value is 0. To do this you simply subtract the mean pixel value from all pixels.

To determine this mean you simply average all images in the whole dataset. If that sounds like a lot of work, we found this to be around 0.25 in the LUNA16 competition.

Warning: Do not zero center with the mean per image (like is done in some kernels on here). The CT scanners are calibrated to return accurate HU measurements. There is no such thing as an image with lower contrast or brightness like in normal pictures.
    '''
    image = image - PIXEL_MEAN
    return image


'''
To save storage space, don't do normalization and zero centering beforehand, but do this online (during training, just after loading). If you don't do this yet, your image are int16's, which are smaller than float32s and easier to compress as well.
'''