import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from skimage.io import ImageCollection
from pprint import pprint
from skimage import io
import numpy as np


def readAllImages(path):
    img_arr = ImageCollection(path + '/*.png')
    return img_arr,len(img_arr)


blank = readAllImages('''Classify 4/Blank_aug''')
border = readAllImages('''Classify 4/Border_aug''')
center = readAllImages('''Classify 4/Center_aug''')

arr1,len1 = blank[0],blank[1]   # 2
arr2,len2 = border[0],border[1]  # 0
arr3,len3 = center[0],center[1] # 1


y1 = np.full((len1,),2)
print(y1.shape)

y2 = np.zeros((len2,))
print(y2.shape)

y3 = np.ones((len3,))
print(y3.shape)

y_test = np.concatenate([y1,y2,y3])
y_test = y_test .reshape((-1,1))
print(y_test.shape)

all_images = []
for i in arr1:
    all_images.append(i)

for i in arr2:
    all_images.append(i)

for i in arr3:
    all_images.append(i)





x_train = np.stack(all_images,axis=0)
print(x_train.shape)


# # 数据预处理函数
# def preprocess(x, y):
#     x = tf.cast(x, dtype=tf.float32) / 255.
#     x = tf.reshape(x, [-1, 32, 32, 1])
#     y = tf.one_hot(y, depth=10)  # one_hot 编码
#     return x, y
#
#
# # 加载数据集
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
#
# # 样本图像周围补0（上下左右均补2个0），将28*28的图像转成32*32的图像
# paddings = tf.constant([[0, 0], [2, 2], [2, 2]])
# x_train = tf.pad(x_train, paddings)
# x_test = tf.pad(x_test, paddings)
#
# train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_db = train_db.shuffle(10000)  # 打乱训练集样本
# train_db = train_db.batch(128)
# train_db = train_db.map(preprocess)
#
# test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# test_db = test_db.shuffle(10000)  # 打乱测试集样本
# test_db = test_db.batch(128)
# test_db = test_db.map(preprocess)
#
# batch = 32
#
# # 创建模型
# model = keras.Sequential([
#     # 卷积层1
#     keras.layers.Conv2D(6, 5),  # 使用6个5*5的卷积核对单通道32*32的图片进行卷积，结果得到6个28*28的特征图
#     keras.layers.MaxPooling2D(pool_size=2, strides=2),  # 对28*28的特征图进行2*2最大池化，得到14*14的特征图
#     keras.layers.ReLU(),  # ReLU激活函数
#     # 卷积层2
#     keras.layers.Conv2D(16, 5),  # 使用16个5*5的卷积核对6通道14*14的图片进行卷积，结果得到16个10*10的特征图
#     keras.layers.MaxPooling2D(pool_size=2, strides=2),  # 对10*10的特征图进行2*2最大池化，得到5*5的特征图
#     keras.layers.ReLU(),  # ReLU激活函数
#     # 卷积层3
#     keras.layers.Conv2D(120, 5),  # 使用120个5*5的卷积核对16通道5*5的图片进行卷积，结果得到120个1*1的特征图
#     keras.layers.ReLU(),  # ReLU激活函数
#     # 将 (None, 1, 1, 120) 的下采样图片拉伸成 (None, 120) 的形状
#     keras.layers.Flatten(),
#     # 全连接层1
#     keras.layers.Dense(84, activation='relu'),  # 120*84
#     # 全连接层2
#     keras.layers.Dense(10, activation='softmax')  # 84*10
# ])
# model.build(input_shape=(batch, 32, 32, 1))
# model.summary()
#
# model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
# # 训练
# history = model.fit(train_db, epochs=50)
#
# # 损失下降曲线
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.show()
#
# # 测试
# model.evaluate(test_db)