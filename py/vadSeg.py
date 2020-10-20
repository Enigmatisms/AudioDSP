"""
    @see ZY_de's commit --- VAD/vad2.m and codes related to that
    @author HQY
    @date 2020.10.20
    静音分割
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa as lr

maxsilence = 10
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
    # TODO: 可能出错
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

def VAD(x, wlen, inc, NIS):
    x = x.T.reshape(-1, 1)
    y = enframe(x, wlen, inc)
    y = y.T
    fn = y.shape[1]
    amp = sum(y ** 2)
    zcr = zeroCrossingRate(y, fn)
    ampth = np.mean(amp[:NIS])
    zcrth = np.mean(zcr[:NIS])
    amp2 = 90 * ampth
    amp1 = 110 * ampth
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
    if x2[e1] == 0:
        print("Error: could not find the ending point.")
        x2[e1] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(e1):
        SF[x1[i]:x2[i]] = 1
        NF[x1[i]:x2[i]] = 0
    speechIndex = np.arange(SF.size)[SF == 1]
    starts, ends = findSegment(speechIndex)
    print("Start size and ends size:", len(starts), " & ", len(ends))
    return np.array(starts), np.array(ends)
        
if __name__ == "__main__":
    number = 1
    file = "..\\voice_stream\\" + str(number) + ".wav"
    data, sr = lr.load(file)



    N = data.size
    time = np.arange(N) / sr

    wlen = 1500
    inc = 700

    IS = 0.5
    overlap = wlen - inc
    NIS = int(np.round((IS * sr - wlen) / inc + 1))
    fn = int(np.round((N - wlen) / inc) + 1)
    frameTime = frameDuration(fn, wlen, inc, sr)

    print("Start to do VAD...")
    starts, ends = VAD(data, wlen, inc, NIS)
    starts *= inc
    ends *= inc
    print("Voice starts:", starts)
    print("Voice ends:", ends)
    print("While the total frame number is: ", fn)
    print("VAD completed.")


    plt.plot(np.arange(data.size), data, c = 'k')

    for i in range(len(starts)):
        ys = np.linspace(-1, 1, 5);
        plt.plot(np.ones_like(ys) * starts[i], ys, c = 'red')
        plt.plot(np.ones_like(ys) * ends[i], ys, c = 'blue')
    
    plt.show()


