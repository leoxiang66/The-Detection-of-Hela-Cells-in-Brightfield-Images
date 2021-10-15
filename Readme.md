# Lenet Implementation For Cell Segmentation

## I. IDEA

1. Divide the preprocessed image into small patches (20*20 size)
2. classify these patches as **blank** or **cell border** or **nucleus**
3. use different colors to mark different types of patches
   -  For example: blank: White; cell border: green; nucleus: red



##### result preview:

<img src="https://i.loli.net/2021/04/02/NGVg9cioL1AZk2q.png" alt="image-20210402083547389"  />





## 2. DATASET

1. blank data
   - number: 714
   - patch size: 20x20
   
     ![image-20210402083943177](https://i.loli.net/2021/04/02/jIxKLBtpAGk7dVm.png)
   
2. border data

   - number: 1658
   - patch size: 20x20

     ![image-20210402084049242](https://i.loli.net/2021/04/02/LRPtCIMfOdmnYHr.png)

3. center data

   - number: 1208
   - patch size: 20x20

     ![image-20210402084132724](https://i.loli.net/2021/04/02/fzdiAonqtuObVve.png)



## 3. IMPLEMENTATION

### 3.1) Data Augmentation

**Transformations we have done**:

- horizontal and vertical shift
- horizontal and vertical flip
- random rotation
- random zoom

1. import libraries

   ```python
   import tensorflow as tf
   from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
   from numpy import expand_dims
   from matplotlib import pyplot
   from skimage import io
   from skimage.io import ImageCollection
   ```

2. load original images

   ```python
   blank_arr = ImageCollection('Blank/*.tif')
   border_arr = ImageCollection('Border/*.tif')
   center_arr = ImageCollection('Center/*.tif')
   ```

3. Function for data augmentation

   ```python
   def da(img,prefix,dest):
       data = img_to_array(img)
       samples = expand_dims(data, 0)
       
       # horizontal shift
       datagen = ImageDataGenerator(width_shift_range=[-1,1])
       it = datagen.flow(samples, batch_size=1)
       for i in range(9):
           batch = it.next()
           image = batch[0].astype('uint8')
           io.imsave(f'dest/{prefix}{i}.tif',image)
           
       # vertical shift
       datagen = ImageDataGenerator(width_shift_range=0.2)
       it = datagen.flow(samples, batch_size=1)
       for i in range(9,18):
           batch = it.next()
           image = batch[0].astype('uint8')
           io.imsave(f'dest/{prefix}{i}.tif',image)
           
       # random rotation
       datagen = ImageDataGenerator(rotation_range=90)
       it = datagen.flow(samples, batch_size=1)
       for i in range(18,27):
           batch = it.next()
           image = batch[0].astype('uint8')
           io.imsave(f'dest/{prefix}{i}.tif',image)
       
       # horizontal and vertical flip
       datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
       it = datagen.flow(samples, batch_size=1)
       for i in range(27,36):
           batch = it.next()
           image = batch[0].astype('uint8')
           io.imsave(f'dest/{prefix}{i}.tif',image)
           
       # random zoom
       datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
       it = datagen.flow(samples, batch_size=1)
       for i in range(36,45):
           batch = it.next()
           image = batch[0].astype('uint8')
           io.imsave(f'dest/{prefix}{i}.tif',image)
   ```

   

4. generate the images

   ```python
   for i in blank_arr:
       da(i,f'Blank_aug_{i}','Blank_aug')
   
   for i in border_arr:
       da(i,f'Border_aug_{i}','Border_aug')
   
   for i in center_arr:
       da(i,f'Center_aug_{i}','Center_aug')
   ```

   

### 3.2) Lenet

1. import libraries

   ```python
   import tensorflow as tf
   import matplotlib.pyplot as plt
   from tensorflow import keras
   from skimage.io import ImageCollection
   from pprint import pprint
   from skimage import io
   import numpy as np
   ```

2. load the dataset

   - load the images as numpy arrays and stack them to a single numpy array
   - define the labels
     - border label = 0
     - center label = 1
     - blank label = 2
   - split the dataset into training dataset and test dataset
   
   ```python
   def readAllImages(path):
       img_arr = ImageCollection(path + '/*.tif')
       return img_arr
   
   # load the data
   arr1 = readAllImages('''dataset/Blank_aug''')
   arr2 = readAllImages('''dataset/Border_aug''')
   arr3 = readAllImages('''dataset/Center_aug''')
   
   # initialize the dataset
   train_images = []
   test_images = []
   train_y = []
   test_y = []
   
   # also split the data into training data and test data
   for i in arr1:
       if i >= 1000:
           test_images.append(i)
           test_y.append(2)
       else:
           train_images.append(i)
           train_y.append(2)
   
   for i in arr2:
       if i >= 1000:
           test_images.append(i)
           test_y.append(0)
       else:
           train_images.append(i)
           train_y.append(0)
           
   for i in arr3:
       if i >= 1000:
           test_images.append(i)
           test_y.append(1)
       else:
           train_images.append(i)
           train_y.append(1)
   
   # build the dataset
   y_train = np.array(train_y,dtype='int32')
   y_test = np.array(test_y,dtype='int32')
   x_train = np.stack(train_images,axis=0)
   x_test = np.stack(test_images,axis=0)
   ```
   
3. preprocessing the data

   - the input image for lenet should be 32x32, whereas our patches have size of 20x20
     - padding the patches
   - normalize the data
   - using one-hot encoding for the labels

   ```python
   # Fill 0s around the image (six 0s on the top, bottom, left and right respectively)
   # 20x20 => 32x32
   paddings = tf.constant([[0,0],[6, 6], [6, 6]])
   x_train = tf.pad(x_train, paddings)
   x_test = tf.pad(x_test, paddings)
   
   def preprocess(x, y):
       x = tf.cast(x, dtype=tf.float32) / 255.
       x = tf.reshape(x, [-1, 32, 32, 1])
       y = tf.one_hot(y, depth=3)  # one_hot encoding
       return x, y
   
   train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
   train_db = train_db.shuffle(10000)  # randomly shuffle the dataset
   train_db = train_db.batch(128)
   train_db = train_db.map(preprocess)
   
   test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
   test_db = test_db.shuffle(10000)  # randomly shuffle the dataset
   test_db = test_db.batch(128)
   test_db = test_db.map(preprocess)
   ```

4. Model initialization

   - model summary: 3 constitutional layers and 2 fully connected layers

     <img src="https://i.loli.net/2021/04/02/mnd9qLBGR6YpFED.png" alt="image-20210402091145126" style="zoom: 80%;" />

     ```python
     batch=32
     model = keras.Sequential([
         keras.layers.Conv2D(6, 5), 
         keras.layers.MaxPooling2D(pool_size=2, strides=2), 
         keras.layers.ReLU(),  
     
         keras.layers.Conv2D(16, 5), 
         keras.layers.MaxPooling2D(pool_size=2, strides=2),  
         keras.layers.ReLU(),  
         
         keras.layers.Conv2D(120, 5),  
         keras.layers.ReLU(), 
         
         keras.layers.Flatten(),
         keras.layers.Dense(84, activation='relu'),  
         keras.layers.Dense(3, activation='softmax') 
     ])
     model.build(input_shape=(batch, 32, 32, 1))
     ```

     

5. compile and fit model

   ```python
   model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=['categorical_accuracy'])
   # training model
   history = model.fit(train_db, epochs=15)
   ```

   ![image-20210402092246241](https://i.loli.net/2021/04/02/SjRUeilOuHY3VJ5.png)

6. predict on test set

   ```python
   model.evaluate(test_db)
   ```

   ![image-20210402092357088](https://i.loli.net/2021/04/02/3j1cmehMEiXgx2n.png)





### 3.3) Visualization

Use different colors to mark different types of patches

-  blank: White
-  cell border: green
-  nucleus: red



Input sample image:

![image-20210402095244512](https://i.loli.net/2021/04/02/yGDxATE83pUwWu9.png)



```python
# 1. load the pretrained model
lenet_model = tf.keras.models.load_model('lenet-model.h5')

# 2. load the input image
input_image = io.imread(...)

# 3. crop the image to patches
delta = 20
results = []
for i in range(26):
    for j in range(26):
         cropped = input_image[i*delta:(i+1)*delta,j*delta:(j+1)*delta]
         results.append(cropped)

patches = np.stack(results,axis=0)

# 4. preprocessing the patches
'''
similar codes in lenet implementation
'''

# 5. predict the patches
results=lenet_model.predict(db,1,verbose =2)
label = [np.argmax(results[i]) for i in range(len(results))]
label_2d = np.array(label).reshape((26,26))
```

![image-20210402100104890](https://i.loli.net/2021/04/02/o25SVqmxAFL4diB.png)

```python
# 6. mark the patches with colors
output_image = np.zeros((520,520,3))
for i in range(26):
    for j in range(26):
         v = labels_2d[i][j]
         
         # blank
         if v == 2:
            output_image[i*20:(i+1)*20,j*20:(j+1)*20] = [255,255,255] # white

         # center
         elif v == 1: output_image[i*20:(i+1)*20,j*20:(j+1)*20] = [255,0,0] # red

         # border
         else: output_image[i*20:(i+1)*20,j*20:(j+1)*20] = [0,128,0]       # green
                
                
# 7. show the image
'''
...
'''
```

![](https://i.loli.net/2021/04/02/mtN4vSBLfG6DR9U.png)

# 8. Official Paper
https://ieeexplore.ieee.org/document/9543103/
