from skimage import io
import numpy as np
# from libtiff import TIFF
from pprint import pprint

# def tiff2Stack(filePath):
#     tif = TIFF.open(filePath,mode='r')
#     stack = []
#     for img in list(tif.iter_images()):
#         stack.append(img)
#     return  stack

img = io.imread('/Users/taoxiang/Downloads/BF-C2DL-HSC/01/t0000.tif' )
rows,cols =  img.shape

for i in range(5000):
    x = np.random.randint(rows)
    y = np.random.randint(cols)
    img[x,y] = 255




io.imshow(img)
io.show()


# img = tiff2Stack('/Users/taoxiang/Downloads/BF-C2DL-HSC/01/t0000.tif')
pprint(img)