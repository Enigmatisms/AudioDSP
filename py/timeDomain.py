#-*-coding:utf-8-*-

import sys
import numpy as np
import pickle as pkl
from VAD import VAD
import matplotlib.pyplot as plt
from featureExtract import windowedFrames, getFeatures

"""
    输入需要分割的数据，进行VAD以及数字识别
    path - 文件路径
    use_forest - 使用随机森林？
"""
def loadAndSeg(path = ".\\main.wav", manual = True, use_forest = True, sr = 22050):
    vad = VAD(path)
    numbers = vad.end2end(manual = manual, do_plot = True)
    test_set = []
    for seg in numbers:
        fr = windowedFrames(seg, 50)
        test_set.append(getFeatures(fr))
    test_set = np.array(test_set)
    if use_forest:
        path = "..\\model\\forest.bin"
    else:
        path = "..\\model\\svm.bin"
    with open(path, "rb") as file:
        clf = pkl.load(file)
    res = clf.predict(test_set)
    print("Predicted result:", res)

if __name__ == "__main__":
    num = 1
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    path = "..\\random%d.wav"%(num)
    loadAndSeg(path)