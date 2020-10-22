"""
    @author HQY
    @date 2020.10.20
    静音分割 数据集处理
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
from vadSeg import *

if __name__ == "__main__":
    wlen = 800
    inc = 400
    IS = 0.5
    for number in range(10):
        for i in range(10):
            print("Start to do Process VAD on number(%d) file(%d)..."%(number, i))
            ipath = "..\\segment\\%d\\%d%02d.wav"%(number, number, i)
            if os.path.exists(ipath) == False:
                print("No file named \'%s\' exists!"%(ipath))
                break
            data, sr = lr.load(ipath)
            N = data.size
            overlap = wlen - inc
            NIS = int(np.round((IS * sr - wlen) / inc + 1))
            fn = int(np.round((N - wlen) / inc) + 1)
            frameTime = frameDuration(fn, wlen, inc, sr)
            reset(2)             # vadSeg中的全局变量
        
            data_t = data.T.reshape(-1, 1)
            y = enframe(data_t, wlen, inc)          # 分帧操作
            y = y.T
            fn = y.shape[1]
            zcrs = zeroCrossingRate(y, fn)
            amps = calculateMeanAmp(y)              # 平均幅度
            # ================= 语音分割以及自适应化 ========================
            starts, ends = VAD(y, zcrs, wlen, inc, NIS)   
            starts, ends = faultsFiltering(amps, starts, ends, 0.005)
            starts = vadPostProcess(amps, starts, 12)
            adaptiveThreshold(amps, starts, ends)
            # =============================================================
    
            plt.figure(1)
        
            plt.plot(np.arange(data.size), data, c = 'k')
            for i in range(len(starts)):
                ys = np.linspace(-1, 1, 5);
                plt.plot(np.ones_like(ys) * starts[i] * inc, ys, c = 'red')
                plt.plot(np.ones_like(ys) * ends[i] * inc, ys, c = 'blue')
            opath = "..\\pics\\whole_%d%02d.jpg"%(number, i)
            plt.savefig(opath)
            plt.cla()
            plt.clf()
            plt.close()
            averageAmpPlot(amps, starts, ends, False, True, "..\\pics\\amp_%d%02d.jpg"%(number, i))            

