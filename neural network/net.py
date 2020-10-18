'''
yk 2020-10-18
基于mfcc特征的语音识别~~~~~~
用简单网络跑只能上70%，因此加入了卷积操作，把mfcc看成图像进行处理。。
最终测试效果还行，80%准确率
'''



import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense,Activation,Dropout,Flatten,Convolution2D,MaxPooling2D
from keras.regularizers import l2

x_train=np.loadtxt("mfcc_data.txt")
y_train=np.loadtxt("mfcc_label.txt")

x_test=np.loadtxt("mfcc_cdata.txt")
y_test=np.loadtxt("mfcc_clabel.txt")
x_train = x_train.reshape(-1,40,13,1)
x_test = x_test.reshape(-1,40,13,1)
model=Sequential([
    Convolution2D
    (
        input_shape=(40,13,1),
        filters=32,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu'
    ),
    MaxPooling2D
    (
        pool_size=2,
        strides=2,
        padding='same'
    ),
    Convolution2D
    (
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu'
    ),
   MaxPooling2D
    (
        pool_size=2,
        strides=2,
        padding='same'
    ),
        Convolution2D
    (
        filters=128,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu'
    ),   
    MaxPooling2D
    (
        pool_size=2,
        strides=1,
        padding='same'
    ),
    Flatten(),
    Dense(units=100,bias_initializer='one',kernel_regularizer=l2(0.01)),
    Activation('relu'),
    Dropout(0.2),
    Dense(units=100,bias_initializer='one',kernel_regularizer=l2(0.01)),
    Activation('relu'),
    Dropout(0.2),
    Dense(units=10,bias_initializer='one',kernel_regularizer=l2(0.01)),
    Activation('softmax')
])

#sgd=SGD(lr=0.00001, decay=1e-6, momentum=0.98, nesterov=True)
adam=Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train,y_train,batch_size=32,epochs=200)

loss,accuracy=model.evaluate(x_train,y_train)

print('\ntrain loss',loss)
print('train accuracy',accuracy)

loss,accuracy=model.evaluate(x_test,y_test)
print('\ntest loss',loss)
print('test accuracy',accuracy)
