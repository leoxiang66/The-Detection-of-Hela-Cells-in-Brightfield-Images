# import packages
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def img_data_augmentation(count,prefix,num_img,suffix):
    datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for i in range(num_img):
        img = load_img(
            f'./source_img/{prefix}{i}.{suffix}')  # this is a PIL image, please replace to your own file path
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory

        i = 1
        for batch in datagen.flow(x, batch_size=1, save_to_dir='./da/', save_prefix='lena',
                                  save_format='jpg'):
            i += 1
            if i > count:
                break  # otherwise the generator would loop indefinitely


img_data_augmentation(10,'lena',1,'png')

