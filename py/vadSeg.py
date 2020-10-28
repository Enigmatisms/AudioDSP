"""
    @see ZY_de's commit --- VAD/vad2.m and codes related to that
    @author HQY
    @date 2020.10.20
    静音分割
    @todo:
        # 如何进行鲁棒的噪音估计？
        # 删除过短帧
    ### @deprecation 本模块即将被删除，因为太乱了，整理优化后的代码放在VAD.py中
"""

import numpy as np
import matplotlib.pyplot as plt
from random import choice as rdc
import librosa as lr
import cv2 as cv
import sys


_fig_num = 1
__maxsilence__ = 10
__minlen__  = 12

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

def VAD(y, zcrs, wlen, inc, NIS):
    fn = y.shape[1]
    amp = sum(y ** 2)
    zcrth = np.mean(zcrs[:NIS])
    amp2 = 0.155                # 如何进行鲁棒的噪音估计？
    amp1 = 0.205
    zcr2 = 0.15 * zcrth

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
            elif amp[n] > amp2 or zcrs[n] < zcr2:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        elif status == 2:
            if amp[n] > amp2 or zcrs[n] < zcr2:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < __maxsilence__:
                    count[xn] += 1
                elif count[xn] < __minlen__:
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
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(e1):
        SF[x1[i]:x2[i]] = 1
        NF[x1[i]:x2[i]] = 0
    speechIndex = np.arange(SF.size)[SF == 1]
    starts, ends = findSegment(speechIndex)
    return np.array(starts), np.array(ends)

# VAD 得到结果后需要进行进一步处理 退火式的搜索
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
    max_val = max(amps)
    if thresh > max_val / 4:
        thresh = max_val / 4
    for start, end in zip(starts, ends):
        if np.mean(amps[start:end]) > thresh and end - start >= __minlen__: # 单门限阈值 长度限制
            _starts.append(start)
            _ends.append(end)
    return _starts, _ends

# 平均幅度的可视化
def averageAmpPlot(amps, starts, ends, plot_diff = True, save = False, path = ""):
    global _fig_num
    plt.figure(_fig_num)
    _fig_num += 1
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
    if save:
        plt.savefig(path)
        plt.cla()
        plt.clf()
        plt.close()

def plotZeroCrossRate(zcrs, starts, ends):
    global _fig_num
    plt.figure(_fig_num)
    _fig_num += 1
    zcrs /= max(zcrs)
    plt.plot(np.arange(zcrs.size), zcrs, c = 'k')
    plt.scatter(np.arange(zcrs.size), zcrs, c = 'k', s = 6)
    for start, end in zip(starts, ends):
        ys = np.linspace(0, 1, 3);
        plt.plot(np.ones_like(ys) * start, ys, c = 'red')
        plt.plot(np.ones_like(ys) * end, ys, c = 'blue')
    plt.title("Average Zero Crossing Rate")


def calculateMeanAmp(frames):
    wlen, fn = frames.shape
    amps = np.zeros(fn)
    for i in range(fn):
        amps[i] = sum(abs(frames[:, i])) / wlen
    return amps

def adaptiveThreshold(amps, starts, ends):
    for i in range(starts.__len__()):
        start = starts[i]
        if start < 0:
            start = 0
        end = ends[i]
        _amp = amps[start:end]
        _amp = _amp / max(_amp) * 255
        _amp = _amp.astype(np.uint8)
        _amp = cv.threshold(_amp, 0, 1, cv.THRESH_BINARY | cv.THRESH_OTSU)[1].astype(int)
        _amp = (_amp[:-1] - _amp[1:])
        _amp.resize(_amp.size + 1)
        end_counter = 0
        for j in range(_amp.size):
            if _amp[j] > 0:
                if end_counter < 2:                         # 多峰时，最多移动两次终止点
                    end_counter += 1
                    ends[i] = start + j     # 精分割，并扩大一帧选区

# 绘制过零率 幅度结合分量
def plotCombined(amps, zcrs, starts, ends, coeff = 0.5, plot_thresh = True, save = False, path = ""):
    to_plot = amps.copy()
    if plot_thresh:
        to_plot.astype(int)
    for i in range(starts.__len__()):
            start = starts[i]
            end = ends[i]
            _amp = amps[start:end]
            _zcr = zcrs[start:end]
            _amp = _amp / max(_amp)
            _zcr = _zcr / max(_zcr) 
            _amp = (coeff * _amp +(1 - coeff) * _zcr)
            if plot_thresh:
                _amp *= 255
                _amp = _amp.astype(np.uint8)
                _amp = cv.threshold(_amp, 0, 1, cv.THRESH_BINARY | cv.THRESH_OTSU)[1].astype(int)
            to_plot[start:end] = _amp.ravel()
    global _fig_num
    plt.figure(_fig_num)
    _fig_num += 1
    plt.plot(np.arange(to_plot.size), to_plot, c = 'k')
    plt.scatter(np.arange(to_plot.size), to_plot, c = 'k', s = 6)
    for i in range(len(starts)):
        ys = np.linspace(0, 0.5, 5);
        plt.plot(np.ones_like(ys) * starts[i], ys, c = 'red')
        plt.plot(np.ones_like(ys) * ends[i], ys, c = 'blue')
    plt.title("ZCR AMP Combined. Coeff = %.2f"%(coeff))
    if save:
        plt.savefig(path)

# 退火搜索，想法是：向左右进行端点搜索，退火温度不要太高
    # 开始时对start - 20, end + 20 内的amps归一化
    # 最大迭代次数为20次：
    # 前6次最大步长为3帧，此后变为两帧（左右）
    # 较高的值按照概率接受（接受变化，但不接受作为最终值）
    # 较低的值分为正常、临界、异常值
        # 正常值（合适幅值）直接移动
        # 临界（处于过大与过小值之间的，难以判定值）：记录临界值数目，按照数目确定的概率接收 0.06~0.03
            # 临界值多了，接受概率就会变低
        # 异常值（过小值 < 0.03）：直接终止算法
    # 帧移不超过20帧（20帧对应了6000）（调整的最大限度）
    # 开始时就小于0.03的端点不优化
def annealingSearch(amps, starts, ends, max_mv = 20):
    length = starts.__len__()
    for i in range(length):
        start = starts[i] - max_mv
        old_end = ends[i]                      # 记录原有的终止点
        end = ends[i] + max_mv
        if i == 0:      # 防止array越界
            start = max(0, start)
        elif i == length - 1:
            end = min(amps.size - 1, end)
        max_val = max(amps[start:end])
        _temp = amps[start:end].copy()
        _temp /= max_val                    # 归一化
        ends[i] = start + annealing(_temp, old_end - start)  # 从原有的终止点开始搜索 返回值为移动的步长

# 异常值阈值 0.03 默认
def annealing(seg, end, max_iter = 30, ab_thresh = 0.03, start_temp = 0.3):
    now_pos = end
    seg_len = len(seg)
    min_pos = now_pos
    min_val = seg[now_pos]
    tipping_cnt = 0
    for i in range(max_iter):
        temp = start_temp / (i + 1)
        if i > 6:
            step = rdc((1, 2)) * rdc((-1, 1))
        elif i > 2:
            step = rdc((1, 2, 3)) * rdc((-1, 1))
        else:
            step = rdc((1, 1, 2, 2, 3))
        tmp_pos = now_pos + step
        if tmp_pos >= seg_len:
            tmp_pos = seg_len - 1
        elif tmp_pos <= 0:
            tmp_pos = 0
        now_val = seg[tmp_pos]
        if now_val < min_val:       # 接受当前移动以及最终移动
            min_val = now_val
            now_pos = tmp_pos
            min_pos = tmp_pos
        else:
            # 一般而言，now_val - min_val 都是 0.01级别的
            # 如果符合metropolis准则，则接受当前移动（但由于比end大，不接受其作为最终解）
            if  np.random.uniform(0, 1) < np.exp(- (now_val - min_val) / temp):
                now_pos = tmp_pos
        this_val = seg[now_pos]
        if this_val < ab_thresh:
            return min_pos
        elif this_val < 2 * ab_thresh:
            if np.random.uniform(0, 1) < np.exp( - tipping_cnt / 2):
                return min_pos
            tipping_cnt += 1
    return min_pos

def normalizedAmp(amps, starts, ends):
    for start, end in zip(starts, ends):
        max_val = max(amps[start - 15:end + 15])
        amps[start - 15:end + 15] /= max_val
    global _fig_num
    plt.figure(_fig_num)
    _fig_num += 1
    plt.plot(np.arange(amps.size), amps, c = 'k')
    plt.scatter(np.arange(amps.size), amps, c = 'k', s = 6)
    for i in range(len(starts)):
        ys = np.linspace(0, 1, 3);
        plt.plot(np.ones_like(ys) * starts[i], ys, c = 'red')
        plt.plot(np.ones_like(ys) * ends[i], ys, c = 'blue')

def reset(num):
    global _fig_num
    global count
    global silence
    _fig_num = num
    count   = [0]
    silence = [0]

if __name__ == "__main__":
    number = sys.argv[1]
    seg_num = sys.argv[2]
    file = "..\\segment\\%s\\%s%02d.wav"%(number, number, int(seg_num))
    data, sr = lr.load(file)

    N = data.size
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
    y = y.T
    fn = y.shape[1]
    zcrs = zeroCrossingRate(y, fn)
    amps = calculateMeanAmp(y)              # 平均幅度
    # ================= 语音分割以及自适应化 ========================
    starts, ends = VAD(y, zcrs, wlen, inc, NIS)   

    # TODO:自适应过滤门限更改
    starts, ends = faultsFiltering(amps, starts, ends, 0.012)
    starts = vadPostProcess(amps, starts, 12)
    adaptiveThreshold(amps, starts, ends)
    annealingSearch(amps, starts, ends)
    # =============================================================

    print("Voice starts:", starts)
    print("Voice ends:", ends)
    print("While the total frame number is: ", fn)
    print("VAD completed.")
    plt.figure(_fig_num)
    _fig_num += 1

    plt.plot(np.arange(data.size), data, c = 'k')

    for i in range(len(starts)):
        ys = np.linspace(-1, 1, 5);
        plt.plot(np.ones_like(ys) * starts[i] * inc, ys, c = 'red')
        plt.plot(np.ones_like(ys) * ends[i] * inc, ys, c = 'blue')

    averageAmpPlot(amps, starts, ends, True)
    normalizedAmp(amps, starts, ends)

    # plotZeroCrossRate(zcrs, starts, ends)
    # plotCombined(amps, zcrs, starts, ends, plot_thresh = True)
    plt.show()


