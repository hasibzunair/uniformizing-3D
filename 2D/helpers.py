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


def threshold_and_normalize_scan (scan):
    scan = scan.astype(np.float32)
    scan [scan < THRESHOLD_LOW] = -1100
    scan [scan > THRESHOLD_HIGH] = 700
    
    # Maximum absolute value of any pixel .
    max_abs = abs (max(THRESHOLD_LOW, THRESHOLD_HIGH, key=abs))
    
    # This will bring values between -1 and 1
    scan /= max_abs
    
    return scan




#MIN_BOUND = -1000 # anything below this we are not interested in 
#MAX_BOUND = 700   # anything above too

def normalize(image):
    '''
    Our values currently range from -3000 to around 2400. Anything above 400 is not interesting to us, as these are simply bones with different radiodensity. Refer to histogram plot
    '''
    
    MIN_BOUND = min(image.flatten())
    MAX_BOUND = max(image.flatten())
    
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
    

def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.
    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


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