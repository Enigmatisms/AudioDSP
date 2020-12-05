'''
    yk  2020-10-23
    DTW函数
    输入为两个mfcc向量(shape值可以不一样)，输出为差异值
    用每一个语音信号提取的mfcc特征跑dtw效果不是很好（不同数字和相同数字之间相差20%）
    估计要每一帧提取mfcc，再进行dtw计算
'''
import librosa as lr
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from dtw import accelerated_dtw, dtw
from numpy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances

class DTW:
    refs = []       # 参考集合
    def __init__(self, sr = 22050):
        self.sr = sr

    def getReference(self):
        print(">>> Start to load reference templates...")
        DTW.refs = [[] for i in range(10)]
        for num in range(10):
            for i in range(6):
                path = "..\\refs\\%d%d.wav"%(num, i)
                wav, _ = lr.load(path, sr = self.sr)
                wav /= max(wav)
                print("%s size:"%(path), wav.size)
                frames = DTW.enframe(wav, 400, 360)
                mfcc = []
                for fr in frames:
                    feats = lr.feature.mfcc(fr, self.sr, n_mfcc = 20)
                    mfcc.append(feats.ravel())
                mfcc = np.array(mfcc)
                DTW.refs[num].append(mfcc)
        print(">>> Referece templates loading process completed.")
        print(">>> Start template matching via DTW & MFCC")

    @staticmethod
    def windowedFrames(data, fnum = 50):
        length = data.size
        flen = int(np.floor(length / 0.9 / fnum))
        frames = np.zeros((fnum, flen))
        shift = int(0.9 * flen)
        # print(frames.shape, flen, shift, data.size)
        for i in range(fnum):
            if i * shift + flen > length:
                frames[i, : length - i * shift] = data[i * shift:]
            else:
                frames[i, :] = data[i * shift: i * shift + flen]
        return frames

    """
        分帧操作
        win - 窗长
        inc - 帧移
    """
    @staticmethod
    def enframe(x, win, inc = 400):
        nx = len(x)
        nf = int(np.round((nx - win + inc) / inc))
        i = 0
        f = np.zeros((nf, win), dtype = np.float64)
        for i in range(nf):
            temp = x[inc * i:inc * i + win]
            f[i, :temp.size] = x[inc * i:inc * i + win].ravel()      # 没有帧移
        return f

    def multiFrameDTW(self, target): 
        frames = DTW.enframe(target, 400, 360)
        mfcc = []
        for frame in frames:
            feats = lr.feature.mfcc(frame, sr = self.sr, n_mfcc = 20).ravel()
            mfcc.append(feats.ravel())
        mfcc = np.array(mfcc)
        proba = np.zeros(10)
        for i, num in enumerate(DTW.refs):      # 按数字分类
            for ref in num:                     # 每个数字有几个标准模板，标准模板存在一定数量的帧
                dist, _, _, _ = accelerated_dtw(ref, mfcc, dist = lambda x, y: norm(x - y, ord = 2))
                proba[i] += dist
        print("Number proba: ", proba)
        return np.argmin(proba)

if __name__ == "__main__":
    sr = 8000
    _dtw = DTW(sr = sr)
    _dtw.getReference()
    truth = []
    result = []
    for num in range(10):
        for i in range(29):
            path = "..\\full\\c%d\\%d.wav"%(num, i)
            if os.path.exists(path) == False:
                break
            wav = lr.load(path, sr)[0]
            wav /= max(wav)                 # 极值归一化
            res = _dtw.multiFrameDTW(wav)
            truth.append(num)
            result.append(res)
            print("Process: number %d, NO.%d"%(num, i))
    truth = np.array(truth)
    result = np.array(result)
    print("Process completed.")
    cnt = sum((truth - result).astype(bool))
    print("Total %d / %d Faults. Ratio for correct recogniton: %f"%(cnt, len(truth), 1 - cnt / len(truth)))