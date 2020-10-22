"""
    @see ZY_de's commit --- VAD/vad2.m and codes related to that
    @author HQY
    @date 2020.10.20
    静音分割
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
import cv2 as cv
import sys

maxsilence = 15
minlen  = 7   

count   = [0]
silence = [0]

def enframe(x, win, inc = 0):
    nx = len(x)
    if type(win) == int:
        length = win
    else:
        length = len(win)
    if inc == 0:
        inc = length
    nf = int(np.round((nx - length + inc) / inc))
    f = np.zeros((nf, length))
    indf = inc * np.arange(nf).reshape(-1, 1)   # 设置每帧在x中的位移量位置
    for i in range(nf):
        temp = x[inc * i:inc * i + length]
        f[i, :temp.size] = x[inc * i:inc * i + length].ravel()      # 没有帧移
        if type(win) != int:
            f[i, :] *= win                      # 加窗操作
    return f

def findSegment(expr):
    if expr[0] == 0:
        voice_index = np.arange(expr.size)[expr == 1]
    else:
        voice_index = expr
    
    starts = [voice_index[0]]
    ends = []
    for i in range(voice_index.size - 1):
        if voice_index[i + 1] - voice_index[i] > 1:
            starts.append(voice_index[i + 1])
            ends.append(voice_index[i])
    ends.append(voice_index[-1])
    return starts, ends

def frameDuration(frameN, frameL, inc, fs):
    return (np.arange(frameN) * inc + frameL / 2) / fs

def zeroCrossingRate(y, fn):
    if y.shape[1] != fn:
        y = y.T
    wlen = y.shape[0]        # 帧长
    zcr = np.zeros(fn)
    delta = 0.0             # 可能的零点漂移（过的不是0，过的是一个很小的值delta）
    for i in range(fn):
        yn = y[:, i]
        ym = np.zeros(wlen)
        for k in range(wlen):
            if yn[k] >= delta:
                ym[k] = yn[k] - delta
            elif yn[k] < - delta:
                ym[k] = yn[k] + delta
            else:
                ym[k] = 0
        zcr[i] = sum(ym[:-1] * ym[1:] < 0)
    return zcr

def VAD(y, wlen, inc, NIS):
    y = y.T
    fn = y.shape[1]
    amp = sum(y ** 2)
    zcr = zeroCrossingRate(y, fn)
    sampling_pos = int(NIS / 2)
    ampth = (np.mean(amp[:sampling_pos]) + np.mean(amp[-sampling_pos:])) / 2 
    # ampth = np.mean(amp[:NIS])
    zcrth = (np.mean(amp[:sampling_pos]) + np.mean(amp[-sampling_pos:])) / 2
    zcrth = np.mean(zcr[:NIS])
    print("Thresholds:", ampth, ", ", zcrth)
    amp2 = 0.09
    amp1 = 0.11
    zcr2 = 0.1 * zcrth

    status  = 0
    xn = 0
    x1 = [0]
    x2 = [0]
    for n in range(fn):
        if status <= 1:
            if amp[n] > amp1:
                x1[xn] = max(n - count[xn] - 1, 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif amp[n] > amp2 or zcr[n] < zcr2:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        elif status == 2:
            if amp[n] > amp2 or zcr[n] < zcr2:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        else:       # status == 3
            status = 0
            xn += 1
            count.append(0)
            silence.append(0)
            x1.append(0)
            x2.append(0)
    
    e1 = len(x1)
    if x1[-1] == 0:
        e1 -= 1
    # if x2[e1] == 0:
    #     print("Error: could not find the ending point.")
    #     x2[e1] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(e1):
        SF[x1[i]:x2[i]] = 1
        NF[x1[i]:x2[i]] = 0
    speechIndex = np.arange(SF.size)[SF == 1]
    print("SF:", SF)
    print("x1:", x1)
    print("x2:", x2)
    print("fn:", fn)
    starts, ends = findSegment(speechIndex)
    print("Start size and ends size:", len(starts), " & ", len(ends))
    return np.array(starts), np.array(ends)

# VAD 得到结果后需要进行进一步处理，向前取 3600 个数据点（在800 / 400 设置下是9个帧）
# 首先需要求差分：平均幅度变化率,如果一直在递减
# 需要退火式的搜索
# end也需要类似搜索方法 由于end较小，需要一个自适应的门限（start 为1/4， end则要根据对应端点平均幅度调整）
def vadPostProcess(amps, starts, prev = 8):
    threshold = amps[starts] / 4            # 1/4 当前门限
    for i, start in enumerate(starts):
        proba = 0.0
        step = 0
        for step in range(prev):
            if step == 2:       # 当step == 2 时开始引入随机拒绝性
                proba = 0.001
            _amp = amps[start - step]
            if _amp < 0.01:
                break
            if _amp > threshold[i]:   # 当大于门限时，step >= 2 有概率停止，step 越大越容易停止
                if step >= 2 and np.random.uniform() < proba:
                    break
                proba *= 1.5
            else:       # 小于门限直接停止
                break
        starts[i] -= step       # 移动端点门限
    return starts

# 过滤错误的分割结果 建议阈值 thresh = 0.005 (说话段平均幅度小于thresh认为是错误的分割)
def faultsFiltering(amps, starts, ends, thresh):
    _starts = []
    _ends = []
    for start, end in zip(starts, ends):
        if np.mean(amps[start:end]) > thresh:            # 单门限阈值
            _starts.append(start)
            _ends.append(end)
    return _starts, _ends

# 平均幅度的可视化
def averageAmpPlot(amps, starts, ends, plot_diff = True):
    plt.figure(2)
    fn = amps.shape[0]
    plt.plot(np.arange(fn), amps, c = 'k')
    plt.scatter(np.arange(fn), amps, c = 'k', s = 6)

    for start, end in zip(starts, ends):
        ys = np.linspace(-0.5, 0.5, 3);
        plt.plot(np.ones_like(ys) * start, ys, c = 'red')
        plt.plot(np.ones_like(ys) * end, ys, c = 'blue')
        amp_mean = np.mean(amps[start:end])
        plt.annotate("%.3f"%(amp_mean), xy = (start, 0.5), xytext = (start, 0.5))
    plt.title("Average Amplitude")

    if plot_diff:
        plt.figure(3)
        diff = amps[:-1] - amps[1:]
        diff.resize(diff.size + 1)
        plt.plot(np.arange(diff.size), diff, c = 'k')
        plt.scatter(np.arange(diff.size), diff, c = 'k', s = 6)
        for start, end in zip(starts, ends):
            ys = np.linspace(-0.5, 0.5, 3);
            plt.plot(np.ones_like(ys) * start, ys, c = 'red')
            plt.plot(np.ones_like(ys) * end, ys, c = 'blue')
            plt.title("Average Amplitude Difference")

def plotNewStartsEnds(starts, ends, inc = 1):
    for start, end in zip(starts, ends):
        ys = np.linspace(-0.5, 0.5, 3);
        plt.plot(np.ones_like(ys) * start * inc, ys, c = 'green')
        plt.plot(np.ones_like(ys) * end * inc, ys, c = 'orange')

def calculateMeanAmp(frames):
    fn, wlen = frames.shape
    amps = np.zeros(fn)
    for i in range(fn):
        amps[i] = sum(abs(frames[i, :])) / wlen
    return amps

def adaptiveThreshold(amps, starts, ends):
    for i in range(starts.__len__()):
        start = starts[i]
        end = ends[i]
        _amp = amps[start:end]
        maxi = max(_amp)
        _amp = _amp / maxi * 255
        _amp = _amp.astype(np.uint8)
        _amp = cv.threshold(_amp, 0, 1, cv.THRESH_BINARY | cv.THRESH_OTSU)[1].astype(int)
        _amp = (_amp[:-1] - _amp[1:])
        _amp.resize(_amp.size + 1)
        start_rev, end_rev = False, False
        for j in range(_amp.size):
            if _amp[j] < 0:
                if start_rev == False:
                    start_rev == True
                    starts[i] += int(j / 2)          # 精分割，并扩大一帧选区
                else:       # 存在可能的多峰
                    raise ValueError("Start is already reset yet being reset again. Possible spikes detected.")
            elif _amp[j] > 0:
                if end_rev ==  False:
                    end_rev == True
                    ends[i] = int((end + start + j) / 2)     # 精分割，并扩大一帧选区
                else:
                    raise ValueError("End is already reset yet being reset again. Possible spikes detected.")

    return starts, ends

        
if __name__ == "__main__":
    number = sys.argv[1]
    file = "..\\voice_stream\\" + str(number) + ".wav"
    data, sr = lr.load(file)

    N = data.size
    time = np.arange(N) / sr

    wlen = 800
    inc = 400

    IS = 0.5
    overlap = wlen - inc
    NIS = int(np.round((IS * sr - wlen) / inc + 1))
    print("NIS is: ", NIS)
    fn = int(np.round((N - wlen) / inc) + 1)
    frameTime = frameDuration(fn, wlen, inc, sr)

    print("Start to do VAD...")

    data_t = data.T.reshape(-1, 1)
    y = enframe(data_t, wlen, inc)          # 分帧操作
    amps = calculateMeanAmp(y)              # 平均幅度

    starts, ends = VAD(y, wlen, inc, NIS)
    starts, ends = faultsFiltering(amps, starts, ends, 0.005)
    starts = vadPostProcess(amps, starts, 12)
    print(starts)
    rev_starts, rev_ends = adaptiveThreshold(amps, starts, ends)
    print(starts)
    print("Voice starts:", starts)
    print("Voice ends:", ends)
    print("While the total frame number is: ", fn)
    print("VAD completed.")
    plt.plot(np.arange(data.size), data, c = 'k')

    for i in range(len(starts)):
        ys = np.linspace(-1, 1, 5);
        plt.plot(np.ones_like(ys) * starts[i] * inc, ys, c = 'red')
        plt.plot(np.ones_like(ys) * ends[i] * inc, ys, c = 'blue')
    plotNewStartsEnds(rev_starts, rev_ends, inc = inc)

    averageAmpPlot(amps, starts, ends, False)
    plotNewStartsEnds(rev_starts, rev_ends)
    
    # plt.figure(3)
    # plt.plot(np.arange(filtered.size), filtered, c = 'k')
    # for i in range(len(starts)):
    #     ys = np.linspace(-1, 1, 3);
    #     plt.plot(np.ones_like(ys) * starts[i], ys, c = 'red')
    #     plt.plot(np.ones_like(ys) * ends[i], ys, c = 'blue')
    plt.show()


