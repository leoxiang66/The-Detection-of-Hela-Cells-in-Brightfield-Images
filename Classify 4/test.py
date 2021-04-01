from skimage.io import ImageCollection
from skimage import io
blank_arr = ImageCollection('Blank/Blank*.png')
img = io.imread('Blank_aug/Blank_aug _0_15.png')
print(blank_arr[5].shape)
print(img.shape)