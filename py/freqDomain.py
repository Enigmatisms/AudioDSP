#-*-coding:utf-8-*-

import sys
import numpy as np
import pickle as pkl
from VAD import VAD
import matplotlib.pyplot as plt
import librosa as lr
from keras.models import load_model

"""
    输入需要分割的数据，进行VAD以及数字识别
    path - 文件路径
    use_forest - 使用随机森林？
"""
def loadAndSeg(path = ".\\main.wav", manual = True, use_forest = True, sr = 22050):
    model = load_model('..\\model\\model.h5')
    vad = VAD(path)
    numbers = vad.end2end(manual = manual, do_plot = True)
    test_set = []
    for seg in numbers:
        # fr = windowedFrames(seg, 50)
        # test_set.append(getFeatures(fr))
        # y,rate=librosa.load("D:\\DSP\\full\\6\\10.wav")
        mfcc_ori = lr.feature.mfcc(seg,n_mfcc=40)
        mfcc = np.reshape(mfcc_ori,(-1,1))[:400]
        data = np.reshape(mfcc,(1,40,10,1))
        test_set.append(np.argmax(model.predict(data)))
    print("Predict result: ", test_set)

if __name__ == "__main__":
    num = 1
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    path = "..\\random%d.wav"%(num)
    loadAndSeg(path)