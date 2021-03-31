from skimage import io
from pprint import pprint


for i in range(32):
    img = io.imread(f'./Classify/Center/Center {i+1}.png')

    cropped_img = img[:, 2:144, :]

    io.imsave(f'./cropped/Center/Center {i+1}.png', cropped_img)