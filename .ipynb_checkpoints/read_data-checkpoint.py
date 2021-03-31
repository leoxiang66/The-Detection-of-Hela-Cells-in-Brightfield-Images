from skimage.io import ImageCollection
import numpy as np




def readAllImages(path):
    img_arr = ImageCollection(path + '/*.png')
    return np.stack(img_arr,axis=0)


blank = readAllImages('''Classify 4/Blank_aug''')
border = readAllImages('''Classify 4/Border_aug''')
center = readAllImages('''Classify 4/Center_aug''')

print(blank.sh)