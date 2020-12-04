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
from dtw import dtw
from numpy.linalg import norm

class DTW:
    refs = []       # 参考集合
    def __init__(self, sr = 22050):
        self.sr = sr

    def getReference(self):
        for num in range(10):
            for i in range(4):
                path = "..\\refs\\%d%d.wav"%(num, i)
                wav, _ = lr.load(path, sr = self.sr)
                wav /= max(wav)
                print("%s size:"%(path), wav.size)
                frames = DTW.enframe(wav, 800, 720)
                mfcc = []
                for fr in frames:
                    feats = lr.feature.mfcc(fr, self.sr, n_mfcc = 20)
                    # print(type(feats), "\n")
                    mfcc.append(feats.ravel())
                    # print("MFCC shape:", mfcc[-1].shape)
                mfcc = np.array(mfcc)
                print("Number %d / %d mfcc shape: "%(num, i), mfcc.shape)
                mfcc.tofile("..\\refs\\%d%d.bin"%(num, i))
        print("Process completed.")

    @staticmethod
    def loadReference(ref_num = 2):
        DTW.refs = [[] for i in range(10)]
        for num in range(10):
            for i in range(ref_num):
                path = "..\\refs\\%d%d.bin"%(num, i)
                DTW.refs[num].append(np.fromfile(path))
        print("Loading completed.")
            

    # 限定输入mfcc1为模板，mfcc2为目标
    @staticmethod
    def old_dtw(mfcc1, mfcc2):
        M, N = len(mfcc1), len(mfcc2)
        #求两个数之间差异
        d=lambda x, y: sum((x - y)**2)         # 二范数
        #初始化矩阵
        cost = np.ones((M, N))
        cost[0, 0] =d (mfcc1[0], mfcc2[0])
        #计算每一点的cost值
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + d(mfcc1[i], mfcc2[0])
        for j in range(1, N):
            cost[0, j] = cost[0, j-1] + d(mfcc1[0], mfcc2[j])

        # 更新权值，选择最小的cost值
        for i in range(1, M):
            for j in range(1, N):
                choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d(mfcc1[i], mfcc2[j])
        return (np.sqrt(cost[-1, -1]) / M)

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
        frames = DTW.enframe(target, 800, 720)
        mfcc = []
        for frame in frames:
            feats = lr.feature.mfcc(frame, sr = self.sr, n_mfcc = 20).ravel()
            mfcc.append(feats.ravel())
            # print("MFCC shape:", mfcc[-1].shape)
        mfcc = np.array(mfcc)
        proba = np.zeros(10)
        for i, num in enumerate(DTW.refs):      # 按数字分类
            for ref in num:                     # 每个数字有几个标准模板，标准模板存在一定数量的帧
                dist, _, _, _ = dtw(ref, mfcc, dist = lambda x, y: norm(x - y, ord = 2))
                proba[i] += dist
        print("Number proba: ", proba)
        return np.argmin(proba)

if __name__ == "__main__":
    _dtw = DTW()
    get_ref = 0
    if len(sys.argv) > 1:
        get_ref = int(sys.argv[1])
    if get_ref:
        _dtw.getReference()
    else:
        DTW.loadReference()
        truth = []
        result = []
        for num in range(10):
            for i in range(29):
                path = "..\\full\\c%d\\%d.wav"%(num, i)
                if os.path.exists(path) == False:
                    break
                wav = lr.load(path)[0]
                wav /= max(wav)                 # 极值归一化
                res = _dtw.multiFrameDTW(wav)
                truth.append(num)
                result.append(res)
                print("Process: number %d, NO.%d"%(num, i))
        truth = np.array(truth)
        result = np.array(result)
        print("Process completed.")
        cnt = sum((truth - result).astype(bool))
        print("Total %d / %d Faults. Ratio: %f"%(cnt, len(truth), cnt / len(truth)))
