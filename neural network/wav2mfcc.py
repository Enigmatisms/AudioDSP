'''
yk 2020-10-29
降低了mfcc的数目和特征数
yk 2020-10-16
写了个提取mfcc的函数，给出输入wav音频即可求出mfcc
由于大部分mfcc的特征数为12-15（少数有10和11个的）左右，选取13个为标准值，多于13的特征舍去，没有13个特征的补0
鄙人代码水平有限--，代码运行时间有点慢，但是结果还是可以出来的~~
2020-10-18
新增onehot transform转化函数，将label值转化为onehot形式，以便于网络进行训练
'''

import os
import numpy as np
import librosa
#将wav文件转化为数据集格式 
#test_x.shape=(length of all .wav ,mfccs)
#test_y.shape=(length of all .wav,1)
#mfcc_x=test_x          shape(length of all .wav ，mfccs)
#mfcc_y=onehot(test_y)  shape(length of all .wav ，10)
def wav2data(filepath,num_mfcc=20,feature=10):
    test_x=None
    test_y=None
    #提取文件中的所有wav文件中的MFCC并将其转化为标准格式
    for j in range(10):
        path=filepath+"{0}\\".format(j)
        length=np.array(os.listdir(path)).shape[0]
        mfccs=np.zeros([length,num_mfcc*feature])  #统一选用13个40维特征向量，如果没有13个则补零
        for i in range(length):
            file=path+"{0}.wav".format(i)
            y,rate=librosa.load(file)
            mfcc = librosa.feature.mfcc(y=y, sr=rate, n_mfcc=num_mfcc)
            mfcc_1=np.reshape(mfcc,[-1,])
            if mfcc_1.shape[0]>=num_mfcc*feature:
                mfccs[i]= mfcc_1[:num_mfcc*feature]
            else:
                mfccs[i]= np.hstack((mfcc_1,np.zeros([num_mfcc*feature-mfcc_1.shape[0],])))
        if test_x is None:
            test_x=mfccs
        else:
            test_x=np.vstack((test_x,mfccs))
        
        if test_y is None:
            test_y=np.zeros([1,length])
        else:
            test_y=np.hstack((test_y,j*np.ones([1,length])))
    return test_x,test_y.T
def onehottransform(data):
    length=data.shape[0]
    onehots=None
    for i in range(length):
        onehot=np.zeros(10)
        position=int(data[i])
        onehot[position]=1
        if onehots is None:
            onehots=onehot
        else:
            onehots=np.vstack((onehots,onehot))
    return onehots
filepath="D:\\DSP\\full\\"
test_x,test_y=wav2data(filepath)
print(test_x.shape,test_y.shape)
test_y=onehottransform(test_y)
np.savetxt("mfcc1_data.txt",test_x)
np.savetxt("mfcc1_label.txt",test_y)