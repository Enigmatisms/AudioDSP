'''
yk 2020-10-16
用keras自己搭了个网络 （过几天用pytorch试试）
不加入正则化准确率95%（应该过拟合了）
加入正则化参数训练完正确率大概70%左右 
尚未测试验证集情况，过拟合应该问题不是很严重
转化为y_train转化为one-hot向量，用cross-entropy和softmax应该准确率会更高
'''



import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense,Activation,Dropout

x_train=np.loadtxt("mfcc_data.txt")
y_train=np.loadtxt("mfcc_label.txt")

model=Sequential([
    Dense(units=100,input_dim=520,bias_initializer='one'),
    Activation('relu'),
    Dropout(0.1),
    Dense(units=100,bias_initializer='one'),
    Activation('relu'),
    Dropout(0.1),
    Dense(units=100,bias_initializer='one'),
    Activation('relu'),
    Dropout(0.1),
    Dense(units=1,bias_initializer='one'),
])

sgd=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

model.fit(x_train,y_train,batch_size=32,epochs=500)

loss,accuracy=model.evaluate(x_train,y_train)

print('\ntrain loss',loss)
print('train accuracy',accuracy)