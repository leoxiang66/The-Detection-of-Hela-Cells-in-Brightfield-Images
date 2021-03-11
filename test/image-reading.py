from skimage import io

from libtiff import TIFF
from pprint import pprint

# def tiff2Stack(filePath):
#     tif = TIFF.open(filePath,mode='r')
#     stack = []
#     for img in list(tif.iter_images()):
#         stack.append(img)
#     return  stack

a = io.imread('/Users/taoxiang/Downloads/BF-C2DL-HSC/01/t0000.tif')
io.imshow(a)


# a = tiff2Stack('/Users/taoxiang/Downloads/BF-C2DL-HSC/01/t0000.tif')
pprint(a)