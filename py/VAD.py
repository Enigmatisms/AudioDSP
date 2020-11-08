"""
    类封装 / 多线程加速
    @author HQY
    @date 2020.10.28
    @todo：双门限法优化可以继续做的：probing操作
        如果语音段长于某个值，起点终点的中点进行探查
        中点幅度过小，说明双门限把两段声音放到同一个划分中了
        比如数字1的100.wav 退火算法无能为力，而probing操作可以重新分割（不过需要进行insert操作，个人不是很喜欢）
        假设全部有误那就是个O（n^2）插入操作
"""

import os
import sys
import numpy as np
import multiprocessing as mtp
import matplotlib.pyplot as plt
from time import time
from librosa import load as lrLoad
from cv2 import THRESH_BINARY, THRESH_OTSU
from cv2 import threshold as cvThreshold
from modules import *
import xlwt
import json

__seg_cnt__ = 0

class VAD:
    _fig_num = 1
    __maxsilence__ = 10
    __minlen__  = 12
    def __init__(self, file, wlen = 800, inc = 400, IS = 0.5):
        self.y = None       # 分帧结果
        self.zcrs = None
        self.amps = None
        self.ends = None
        self.starts = None
        if not os.path.exists(file):
            raise ValueError("No such file as '%s'!"%(file))
        start_t = time()
        self.data, self.sr = lrLoad(file)
        end_t = time()
        print("Data loaded. Time consumed: ", end_t - start_t)
        # 需要在此处load
        # 每个类对应处理一段语音

        self.N = self.data.size
        self.wlen = wlen
        self.inc = inc
        self.IS = IS
        self.NIS = int(np.round((IS * self.sr - wlen) / inc + 1))
        self.fn = int(np.round((self.N - wlen) / inc))

    # VAD 得到结果后需要进行进一步处理 退火式的搜索
    def _vadPostProcess_(self, prev = 8):
        threshold = self.amps[self.starts] / 4            # 1/4 当前门限
        for i, start in enumerate(self.starts):
            proba = 0.0
            step = 0
            for step in range(prev):
                if step == 2:       # 当step == 2 时开始引入随机拒绝性
                    proba = 0.001
                _amp = self.amps[start - step]
                if _amp < 0.01:
                    break
                if _amp > threshold[i]:   # 当大于门限时，step >= 2 有概率停止，step 越大越容易停止
                    if step >= 2 and np.random.uniform() < proba:
                        break
                    proba *= 1.5
                else:       # 小于门限直接停止
                    break
            self.starts[i] -= step       # 移动端点门限
            if self.starts[i] < 0:
                self.starts[i] = 0

    # 平均幅度的可视化
    @staticmethod
    def _averageAmpPlot_(amps, starts, ends, plot_diff = True, save = False, path = ""):
        plt.figure(VAD._fig_num)
        VAD._fig_num += 1
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

    # @jit(nopython = True)
    @staticmethod
    def _calculateMeanAmp_(y):
        wlen, fn = y.shape
        amps = np.zeros(fn)
        for i in range(fn):
            amps[i] = np.sum(np.abs(y[:, i])) / wlen
        return amps

    def _adaptiveThreshold_(self):
        for i in range(self.starts.__len__()):
            start = self.starts[i]
            if start < 0:
                start = 0
            end = self.ends[i]
            _amp = self.amps[start:end]
            _amp = _amp / max(_amp) * 255
            _amp = _amp.astype(np.uint8)
            _amp = cvThreshold(_amp, 0, 1, THRESH_BINARY | THRESH_OTSU)[1].astype(int)
            _amp = (_amp[:-1] - _amp[1:])
            _amp.resize(_amp.size + 1)
            end_counter = 0
            for j in range(_amp.size):
                if _amp[j] > 0:
                    if end_counter < 2:                         # 多峰时，最多移动两次终止点
                        end_counter += 1
                        self.ends[i] = start + j     # 精分割，并扩大一帧选区

    """
    退火搜索，想法是：向左右进行端点搜索，退火温度不要太高
        开始时对start - 20, end + 20 内的amps归一化
        最大迭代次数为20次：
        前6次最大步长为3帧，此后变为两帧（左右）
        较高的值按照概率接受（接受变化，但不接受作为最终值）
        较低的值分为正常、临界、异常值
            正常值（合适幅值）直接移动
            临界（处于过大与过小值之间的，难以判定值）：记录临界值数目，按照数目确定的概率接收 0.06~0.03
                临界值多了，接受概率就会变低
            异常值（过小值 < 0.03）：直接终止算法
        帧移不超过20帧（20帧对应了6000）（调整的最大限度）
        开始时就小于0.03的端点不优化
    """
    def _annealingSearch_(self, max_mv = 20, min_inv_step = 25):
        length = self.starts.__len__()
        for i in range(length):
            start = self.starts[i] - max_mv
            old_end = self.ends[i]                      # 记录原有的终止点
            end = self.ends[i] + max_mv
            if i == 0:      # 防止array越界
                start = max(0, start)
            elif i == length - 1:
                end = min(self.amps.size - 1, end)
            max_val = max(self.amps[start:end])
            _temp = self.amps[start:end].copy()
            _temp /= max_val                    # 归一化
            if self.ends[i] - self.starts[i] >= min_inv_step:     # 足够长就可以反向退火搜索
                pos, reverse = invAnnealing(_temp, old_end - start, start_temp = 1.2)
                if reverse == True:
                    self.ends[i] = start + pos
                    continue
            self.ends[i] = start + annealing(_temp, old_end - start)  # 从原有的终止点开始搜索 返回值为移动的步长

    @staticmethod
    def _normalizedAmp_(amps, starts, ends):
        for start, end in zip(starts, ends):
            max_val = max(amps[start - min(15, start):end + 15])
            amps[start - min(15, start):end + 15] /= max_val
        plt.figure(VAD._fig_num)
        VAD._fig_num += 1
        plt.plot(np.arange(amps.size), amps, c = 'k')
        plt.scatter(np.arange(amps.size), amps, c = 'k', s = 6)
        for i in range(len(starts)):
            ys = np.linspace(0, 1, 3);
            plt.plot(np.ones_like(ys) * starts[i], ys, c = 'red')
            plt.plot(np.ones_like(ys) * ends[i], ys, c = 'blue')

    def reset(self, num = 1):
        VAD._fig_num = num

    def process(self, do_plot = True, aux = False):
        self.y = enframe(self.data, self.wlen, self.inc)          # 分帧操作
        self.zcrs = zeroCrossingRate(self.y, self.fn)
        self.amps = calculateMeanAmp(self.y)              # 平均幅度
        # ================= 语音分割以及自适应化 ========================
        # self._vadSegment_()   
        self.starts, self.ends = vadSegment(self.y, self.zcrs, self.fn, self.NIS)
        self.starts, self.ends = faultsFiltering(self.amps, self.starts, self.ends, 0.012)
        self._vadPostProcess_(12)
        self._adaptiveThreshold_()
        self._annealingSearch_()
        # =============================================================
        if do_plot:
            print("Voice starts:", self.starts)
            print("Voice ends:", self.ends)
            print("VAD completed.")
            plt.figure(VAD._fig_num)
            VAD._fig_num += 1
            plt.plot(np.arange(self.N), self.data, c = 'k')
            for i in range(len(self.starts)):
                ys = np.linspace(-1, 1, 5);
                plt.plot(np.ones_like(ys) * self.starts[i] * self.inc, ys, c = 'red')
                plt.plot(np.ones_like(ys) * self.ends[i] * self.inc, ys, c = 'blue')
                pos = (self.ends[i] + self.starts[i]) / 2 * self.inc
                segs = self.ends[i] - self.starts[i]
                plt.annotate("%d"%(i), xy = (pos, 1), xytext = (pos, 1))
                plt.annotate("%d"%(segs), xy = (pos, 0.8), xytext = (pos, 0.8))
            if aux:
                VAD._averageAmpPlot_(self.amps, self.starts, self.ends, True)
                # print(self.amps.shape)
                VAD._normalizedAmp_(self.amps, self.starts, self.ends)
            plt.show()
    
    # 还需要一个输入参数：一个xlwt对象，workbook
    def save(self, sheet):
        global __seg_cnt__
        segs = input("False segments:")
        if segs == '':
            print("No false segment. Exiting...")
        else:
            res = set()
            spl = segs.split(',')
            for num in spl:
                temp = num.strip()
                res.add(int(temp))
            for i in range(len(self.starts)):
                if i in res:
                    continue
                start = self.starts[i] * self.inc
                end = self.ends[i] * self.inc
                data = self.data[start:end]
                for j in range(data.size):
                    sheet.write(j + 1, __seg_cnt__, float(data[j]))
                __seg_cnt__ += 1

def vadLoadAndProcess(path, *args, do_plot = False, aux = False):
    vad = VAD(path)
    vad.process(do_plot, aux)
    if len(args) > 0:
        vad.save(args[0])
    
if __name__ == "__main__":
    number = sys.argv[1]
    using_thread = 0
    step_out = False
    to_select = 0
    if sys.argv.__len__() == 3:
        using_thread = int(sys.argv[2])
    elif sys.argv.__len__() == 4:
        to_select = int(sys.argv[3])
    if step_out == False:
        start_t = time()
        if using_thread:
            proc_pool = []
            for i in range(7):
                file = "..\\segment\\%s\\%s%02d.wav"%(number, number, i)
                pr = mtp.Process(target = vadLoadAndProcess, args = (file, False))
                proc_pool.append(pr)
                pr.start()
            file = "..\\segment\\%s\\%s%02d.wav"%(number, number, 7)
            vadLoadAndProcess(file)
            for i in range(7):
                proc_pool[i].join()
        else:
            file = "..\\segment\\%s\\%s%02d.wav"%(number, number, to_select)
            vadLoadAndProcess(file, do_plot = True, aux = True)
        end_t = time()
        print("Running time: ", end_t - start_t)
    else:
        wb = xlwt.Workbook()
        sheet = wb.add_sheet("Sheet1")
        path = "..\\segment\\test.xls"
        seg_number = 5
        for number in range(2):
            for seg in range(seg_number):
                file = "..\\segment\\%s\\%s%02d.wav"%(number, number, seg)
                # 视觉辅助：自动标号
                # 命令行输出：有错的就不进行输出了
                # 需要有自动保存能力
                vadLoadAndProcess(file, sheet, do_plot = True)
        wb.save(path)

