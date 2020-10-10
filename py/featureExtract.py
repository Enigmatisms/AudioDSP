"""
    尝试提取时域特征
    尝试python 并行计算
        由于每一帧特征提取得到的结果都是相互独立的
    个人觉得特征可以使用如下方式组织：
        1. 每帧的短时幅值 比如我们固定地对某一语音信号分成56帧，那么前两个特征共112阶，只需要一个16阶LPC就可以得到最后的特征
        2. 每帧的短时过零率
        3. 每帧的LPC预测系数
        每一个特征向量长128，最好不要超过256，否则个人认为分类器负担太重
    首先是分割操作，个人先随便设一个阈值吧
    对分割好的纯语音信号进行分帧加窗，帧移大概为10%
    此后进行特征提取，放入分类器进行训练
    测试集进行验证
"""

import librosa as lr
from librosa.core.audio import resample
from librosa.core.spectrum import amplitude_to_db
from librosa.util.utils import frame
import numpy as np
import matplotlib.pyplot as plt
import time

# 无加窗操作的framing
def windowedFrames(data, fnum = 56):
    length = data.size
    flen = int(np.floor(length / 0.9 / fnum))
    frames = np.zeros((fnum, flen))
    shift = int(0.9 * flen)
    for i in range(fnum):
        frames[i, :] = data[i * shift: i * shift + flen]
    return frames
    
# 得到一个fnum * 1维向量，为每一帧的平均幅度
def averageAmplitude(frames, fnum = 50):
    res = np.zeros(fnum)
    for i in range(fnum):
        res[i] = np.mean(abs(frames[i, :]))
    res /= np.max(res)      # 最值归一化
    return res

# 平均过零率 向量
def averageZeroCrossingRate(frames, fnum = 50):
    res = np.zeros(fnum)
    _, cols = frames.shape
    for i in range(fnum):
        cnt = 0
        for j in range(cols - 1):
            if (frames[i][j] > 0 and frames[i][j + 1] < 0 ) or (frames[i][j] < 0 and frames[i][j + 1] > 0):
                cnt += 1
        res[i] = cnt / frames.size
    return res

# 平均幅度差 (半帧)
def averageAmpDiff(frames, fnum = 50):
    _, cols = frames.shape
    half = int(cols / 2)
    res = np.zeros(fnum)
    for i in range(fnum):
        res[i] = sum(abs(frames[i, :-half] - frames[i, half:]))
    res / np.max(res)
    return res

# 个人觉得这个特征很可能不行，而且还会很慢
def getLPC(frames, fnum = 50):
    step = int(fnum / 10)
    res = np.zeros(fnum)
    for i in range(0, 10):
        _lpc = lr.core.lpc(frames[i * step], 5)[1:]
        res[i * 5 : (i + 1) * 5] = _lpc
    res /= np.max(res)
    return res

# 如果是一段语音分为50帧，那么得到的结果就是一个长度为200的向量，大概为14 * 14 ~ 15 * 15图片的规模（不过是浮点数）
def getFeatures(frames, fnum = 50):
    amp = averageAmplitude(frames, fnum)
    azc = averageZeroCrossingRate(frames, fnum)
    aad = averageAmpDiff(frames, fnum)
    lpc = getLPC(frames, fnum)
    return np.concatenate((amp, azc, aad, lpc))

if __name__ == "__main__":
    number = 1
    fnum = 50
    feats = None
    for i in range(12):
        path = "..\\simple\\%d\\%d_%02d.wav"%(number, number, i + 1)
        data, sr = lr.load(path, sr = None)
        data = lr.resample(data, sr, 8000)
        frms = windowedFrames(data, fnum)
        if i == 0:
            feats = getFeatures(frms, fnum)
        else:
            _f = getFeatures(frms, fnum)
            feats = np.vstack((feats, _f))
        print("Loading from " + path)
    print("Features: ", feats)
    # plt.plot(np.arange(data.size), data, c = "k")
    # plt.show()