import matplotlib.pyplot as plt
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import pylab
import cv2
import helpers as H
import os

'''
Fancy program which shows the CT scan slice by slice
'''

#Root directory of the project
ROOT_DIR = os.path.abspath(".")
TRAIN_DATASET_PATH = os.path.join(ROOT_DIR, "dataset")
#train_data_paths = glob.glob(os.path.join(TRAIN_DATASET_PATH,'img_datas_1','*.npy'))
train_data_paths = os.listdir(os.path.join(TRAIN_DATASET_PATH, "img_datas_1"))


# the CT scan from TB folder
image_number = "1"

# /home/hasib/imageclef
# /home/hasib/imageclef/TB/training/High/

img_3d = np.load("{}/img_datas_1/{}.npy".format(TRAIN_DATASET_PATH, image_number))
img_3d = H.normalize(img_3d)
print(img_3d.shape)

counter = 0
window = np.array((1,2))

# p and l to control slices
while True:
    if counter >= img_3d.shape[-1]:
        break
    window = img_3d[:,:,counter]
    cv2.imshow("image 3d".format(counter), window)
    
    k = cv2.waitKey(1) & 0xff
    if k != 255: 
        if k == 112:
            print("forward")
            counter+=1
            # save images
            #cv2.imwrite("{}.jpg".format(counter), window )
            print(counter)
            
        if k == 108:
            print("backward")
            counter-=1
            print(counter)

    if k == 27: 
        break

print("All slices shown")
cv2.destroyAllWindows()
