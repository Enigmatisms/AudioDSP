'''
    yk    2020-10-26
    用网络模型，测试数字被识别为其他数字的概率。
    0和6频域上分不开，准确率只有50%
    其他的数字效果还比较好，能达到85以上
    详细准确率请看num_accuracy_result
    图标请看num_accuracy_graph
'''


import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
#读取模型
model=load_model('model.h5')
#读取数据
x_test=np.loadtxt("mfcc_cdata.txt")
y_test=np.loadtxt("mfcc_clabel.txt")
x_test = x_test.reshape(-1,40,13,1)
#读取各个数的长度
length=np.zeros(10,dtype=np.int32)
#记录每个数的loss与accuracy
loss=np.zeros(10)
accuracy=np.zeros(10)
for i in range (len(y_test)):
    length[int(np.where(y_test[i]==1)[0])]+=1
length_c=length.copy()
for i in range(10):   
    if i==0:
        loss[i],accuracy[i]=model.evaluate(x_test[:length[i]-1],y_test[:length[i]-1])
    else:
        length[i]+=length[i-1]
        loss[i],accuracy[i]=model.evaluate(x_test[length[i-1]-1:length[i]-1],y_test[length[i-1]-1:length[i]-1])
        
for i in range(10):      
    print('\n num {0} loss:'.format(i),loss[i],'test accuracy',accuracy[i])

#计算数字被识别为另一个数字的概率
detail=np.zeros([10,10])
for i in range(len(y_test)):
    pre=np.argmax(model.predict(np.reshape(x_test[i],(1,40,13,1))))
    #if pre!=int(np.where(y_test[i]==1)[0]):
    detail[int(np.where(y_test[i]==1)[0])][pre]+=1
for i in range(10):
    detail[i]/=length_c[i]
    print("\n数字{0}被分为0-9的概率为".format(i));print(detail[i])
#结果图像化
plt.imshow(detail,cmap='gray')
plt.ylabel("true label")
plt.xlabel("predict label")