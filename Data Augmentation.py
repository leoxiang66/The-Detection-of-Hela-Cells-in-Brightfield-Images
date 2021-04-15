import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from numpy import expand_dims
from matplotlib import pyplot
from skimage import io
from skimage.io import ImageCollection

blank_arr = ImageCollection('filtered data/blank/*.tif')
border_arr = ImageCollection('filtered data/border/*.tif')
center_arr = ImageCollection('filtered data/center/*.tif')

def da(img,prefix,dest):
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    
    # horizontal shift
    datagen = ImageDataGenerator(width_shift_range=[-1,1])
    it = datagen.flow(samples, batch_size=1)
    for i in range(9):
        batch = it.next()
        image = batch[0].astype('uint8')
        io.imsave(f'{dest}/{prefix}{i}.tif',image)
        
    # vertical shift
    datagen = ImageDataGenerator(width_shift_range=0.2)
    it = datagen.flow(samples, batch_size=1)
    for i in range(9,18):
        batch = it.next()
        image = batch[0].astype('uint8')
        io.imsave(f'{dest}/{prefix}{i}.tif',image)
        
    # random rotation
    datagen = ImageDataGenerator(rotation_range=90)
    it = datagen.flow(samples, batch_size=1)
    for i in range(18,27):
        batch = it.next()
        image = batch[0].astype('uint8')
        io.imsave(f'{dest}/{prefix}{i}.tif',image)
    
    # horizontal and vertical flip
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    it = datagen.flow(samples, batch_size=1)
    for i in range(27,36):
        batch = it.next()
        image = batch[0].astype('uint8')
        io.imsave(f'{dest}/{prefix}{i}.tif',image)
        
    # random zoom
    datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
    it = datagen.flow(samples, batch_size=1)
    for i in range(36,45):
        batch = it.next()
        image = batch[0].astype('uint8')
        io.imsave(f'{dest}/{prefix}{i}.tif',image)
        
        
for i,img in enumerate(blank_arr):
    da(img,f'Blank_aug_{i}','filtered data/Blank_aug')

for i,img in enumerate(border_arr):
    da(img,f'Border_aug_{i}','filtered data/Border_aug')

for i,img in enumerate(center_arr):
    da(img,f'Center_aug_{i}','filtered data/Center_aug')
