from skimage import io, transform
import tensorflow as tf
import numpy as np



# model
lenet_model = tf.keras.models.load_model('lenet-model.h5')
lenet_model.summary()

def showImg(img):
    io.imshow(img)
    io.show()

def visualize(input_image):



    # resize input image
    input_image = transform.resize(input_image,(520,520))[:,:,:3]
    # crop images to patches
    delta = 20
    results = []
    for i in range(26):
        for j in range(26):
            results.append(input_image[i*delta:(i+1)*delta,j*delta:(j+1)*delta,:])


    patches = np.stack(results,axis=0)

    # 样本图像周围补0（上下左右均补6个0），将20*20的图像转成32*32的图像
    paddings = tf.constant([[0, 0], [6, 6], [6, 6], [0, 0]])
    patches = tf.pad(patches,paddings)

    # preprocessing
    patches = tf.cast(patches,dtype=tf.float32)/255.
    patches = tf.reshape(patches,[-1,32,32,3])

    c=lenet_model.predict(patches,verbose =2)
    print(len(c))
    print(c[0])
    print(np.argmax(c[0]))

    tmp = [np.argmax(c[i]) for i in range(len(c))]
    tmp_np = np.array(tmp).reshape((26,26))
    print(tmp_np)



input_image = io.imread('input images/1.png')
visualize(input_image)
io.imshow(input_image)
io.show()