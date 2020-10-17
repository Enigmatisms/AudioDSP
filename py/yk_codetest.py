'''
yk 2020-10-15 
主要是为了尝试一下上传代码^_^
写了一些预加重函数和加窗函数分帧函数
'''

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


#预加重
def add_weight(x,a=0.98):
    len=x.shape[0]
    y=np.zeros((len,))
    for i in range(1,len):
        y[i]=x[i]-a*x[i-1]
    return y

#加窗函数(method=1 矩形窗 method=2 汉明窗 method=3 海宁窗)
def func_w(len_frame,method):
    y=np.zeros((len_frame,1))
    if method==1:
        for n in range(0,len_frame):
            y[n]=1
        return y
    if method==2:
        for n in range(0,len_frame):
            y[n]=(0.54-0.46*np.cos(2*np.pi*n/(len_frame-1)))
        return y
    if method==3:
        for n in range(0,len_frame):
            y[n]=0.5*(1-np.cos(2*np.pi*n/(len_frame-1)))
        return y
#加窗处理
def add_window(x,method=1):
    num_frames=x.shape[0]
    len_frame=x.shape[1]
    y=np.zeros(x.shape)
    xx=np.zeros([len_frame,1])
    w=func_w(len_frame,1)
    for i in range(num_frames):
        xx=np.reshape(x[i,:len_frame],[-1,1]).T
        y[i]=w[:len_frame].T*xx
    return y
#分帧函数
def windowFrames(x,duration,frame = 50):
    length = x.size
    num_frames=int(frame*duration)                 #一秒50帧，求总帧数
    len_frame = int(np.floor(length / 0.9 / num_frames))#求一帧的长度
    frames = np.zeros((num_frames, len_frame))
    shift = int(0.9 * len_frame)
    for i in range(num_frames):
        if (i * shift + len_frame) > length:
            frames[i, : length - i * shift] = x[i * shift:]
        else:
            frames[i, :] = x[i * shift: i * shift + len_frame]
    return frames

if __name__=="__main__":
    frame=50
    y,sr=librosa.load("D:\\DSP\\data\\0.wav")
    duration=librosa.get_duration(y)
    #librosa.display.waveplot(y)

    y_1=add_weight(y)
    #librosa.display.waveplot(y_1)
    y_2=windowFrames(y_1,duration,frame)
    y_3=add_window(y_2,2)
    librosa.display.waveplot(np.reshape(y_3[10:20],[-1,]))