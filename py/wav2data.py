'''
yk 2020-10-16
写了个提取mfcc的函数，给出输入wav音频即可求出mfcc
由于大部分mfcc的特征数为12-15（少数有10和11个的）左右，选取13个为标准值，多于13的特征舍去，没有13个特征的补0
鄙人代码水平有限--，代码运行时间有点慢，但是结果还是可以出来的~~
'''


import os
import numpy as np

#将wav文件转化为数据集格式 
#test_x.shape=(length of all .wav ,mfccs)
#test_y.shape=(1,length of all .wav)
def wav2data(filepath):
    test_x=None
    test_y=None
    #提取文件中的所有wav文件中的MFCC并将其转化为标准格式
    for j in range(10):
        path=filepath+"{0}\\".format(j)
        length=np.array(os.listdir(path)).shape[0]
        mfccs=np.zeros([length,40*13])  #统一选用13个40维特征向量，如果没有13个则补零
        for i in range(length):
            file=path+"{0}.wav".format(i)
            y,rate=librosa.load(file)
            mfcc = librosa.feature.mfcc(y=y, sr=rate, n_mfcc=40)
            mfcc_1=np.reshape(mfcc,[-1,])
            if mfcc_1.shape[0]>=40*13:
                mfccs[i]= mfcc_1[:40*13]
            else:
                mfccs[i]= np.hstack((mfcc_1,np.zeros([40*13-mfcc_1.shape[0],])))
        if test_x is None:
            test_x=mfccs
        else:
            test_x=np.vstack((test_x,mfccs))
        
        if test_y is None:
            test_y=np.zeros([1,length])
        else:
            test_y=np.hstack((test_y,j*np.ones([1,length])))
    return test_x,test_y

if __name__=='__main__':
    filepath="D:\\DSP\\full\\"
    test_x,test_y=wav2data(filepath)
    print(test_x.shape,test_y.shape)
    np.savetxt("mfcc_data.txt",test_x)
    np.savetxt("mfcc_label.txt",test_y)