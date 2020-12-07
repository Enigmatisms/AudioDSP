'''
    yk  2020-10-23
    hqy 2020-12-5
    DTW函数
'''
import librosa as lr
import numpy as np
import os
import matplotlib.pyplot as plt
from dtw import accelerated_dtw, dtw
from numpy.linalg import norm

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
        mfcc = DTW.getMFCC(target, self.sr)
        proba = np.zeros(10)
        for i, num in enumerate(DTW.refs):      # 按数字分类
            for ref in num:                     # 每个数字有几个标准模板，标准模板存在一定数量的帧
                dist, _, _, _ = accelerated_dtw(ref, mfcc, dist = lambda x, y: norm(x - y, ord = 2))
                proba[i] += dist
        print("Number proba: ", proba)
        return np.argmin(proba)

    @staticmethod
    def getMFCC(series, sr, flen = 400, inc = 360):
        frames = DTW.enframe(series, flen, inc)
        mfcc = []
        for frame in frames:
            feats = lr.feature.mfcc(frame, sr = sr, n_mfcc = 20).ravel()
            mfcc.append(feats.ravel())
        mfcc = np.array(mfcc)
        return mfcc

    # referenc 与 target 的匹配过程输出
    @staticmethod
    def dtwDrawPath(ref, tar, sr = 8000):
        mfcc1 = DTW.getMFCC(ref, sr, 200, 180)
        mfcc2 = DTW.getMFCC(tar, sr, 200, 180)
        print("Reference shape:", mfcc1.shape)
        print("Target shape:", mfcc2.shape)
        dist, costs, accost, path = accelerated_dtw(mfcc1, mfcc2, dist = lambda x, y: norm(x - y, ord = 2))
        print("Distance:", dist)
        print("Costs:", costs.shape)
        print("Accumulated costs:", accost.shape)
        print("Matching path:", path)
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 16
        plt.imshow(accost, interpolation = 'nearest', cmap = 'bone', origin = 'lower')
        plt.plot(path[1],path[0], c = 'r', label = '匹配路径')
        plt.ylabel("匹配模板")
        plt.xlabel("待匹配目标")
        plt.colorbar()
        plt.title("累计误差矩阵匹配路径")
        plt.legend()
        plt.show()
        # print(y)

def main():
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

def demoTest(sr = 8000):
    wav1, _ = lr.load("..\\refs\\40.wav", sr = sr)
    wav2, _ = lr.load("..\\full\\c0\\7.wav", sr = sr)
    DTW.dtwDrawPath(wav1, wav2, sr)
    

if __name__ == "__main__":
    # demoTest()
    main()