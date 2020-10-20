#-*-coding:utf-8-*-
"""
    声音信号压缩 加噪以及降噪
    高斯白噪声 时域Kalman滤波
"""

import librosa as lr
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import time
import wave

path_prefix = "..\\raw\\"

class KalmanAudio:
    def __init__(self, _path, order = 12, length = 500, scale = 0.02):
        self.ord = order
        self.frames = None          # 二维分帧后的矩阵 分帧后以一行为一帧
        self.frame_num = 0
        self.len = length
        self.origin_data = None
        self.scale = scale
        self.loadCompressClip(_path)
        self.NC = np.zeros((order, 1))
        self.H = np.zeros((1, order))
        self.NC[-1] = 1.0
        self.H[0, -1] = 1.0

        self.output = self.frames.ravel().copy()
        self.data_size = self.output.size

        #================ KalmanFilter 模块 ================
        self.oldState = np.zeros((order, 1))             # 历史状态
        self.statePre = np.zeros((order, 1))             # 先验估计
        self.statePost = np.zeros((order, 1))            # 后验估计
        self.stateCovPre = np.eye(order) * 0.75 + np.ones((order, order)) * 0.25            # 状态先验协方差
        self.stateCovPost = np.eye(order) * 0.75 + np.ones((order, order)) * 0.25            # 状态后验协方差
        self.stateNoiseErrorCov = 0   # 状态噪声方差
        self.measureNoiseCov = self.scale ** 2  # 观测协方差

    # librosa函数的提取以及修改（让其可以返回协方差）
    @staticmethod
    # @jit(nopython=True)
    def LPC(y, order):
        dtype = y.dtype.type
        ar_coeffs = np.zeros(order + 1, dtype=dtype)
        ar_coeffs[0] = dtype(1)
        ar_coeffs_prev = np.zeros(order + 1, dtype=dtype)
        ar_coeffs_prev[0] = dtype(1)
        fwd_pred_error = y[1:]
        bwd_pred_error = y[:-1]
        den = np.dot(fwd_pred_error, fwd_pred_error) + np.dot(
            bwd_pred_error, bwd_pred_error
        )
        for i in range(order):
            if den <= 0:
                raise FloatingPointError("Numerical error, input ill-conditioned?")
            reflect_coeff = dtype(-2) * np.dot(bwd_pred_error, fwd_pred_error) / dtype(den)
            ar_coeffs_prev, ar_coeffs = ar_coeffs, ar_coeffs_prev
            for j in range(1, i + 2):
                ar_coeffs[j] = ar_coeffs_prev[j] + reflect_coeff * ar_coeffs_prev[i - j + 1]
            fwd_pred_error_tmp = fwd_pred_error
            fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error
            bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp

            q = dtype(1) - reflect_coeff ** 2
            den = q * den - bwd_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2
            fwd_pred_error = fwd_pred_error[1:]
            bwd_pred_error = bwd_pred_error[:-1]
        return (ar_coeffs, np.var(fwd_pred_error))

    def loadCompressClip(self, path:str):
        _data, sr = lr.load(path)
        print("Origin data size: %d"%(_data.size))
        data = lr.resample(_data, sr, 4000)
        sz = data.size
        tail = sz % self.len
        data = data[:-tail]
        self.origin_data = data.copy()                      # 保存原数据
        data += np.random.normal(0, self.scale, data.size)       # 加噪处理，高斯白噪声 标准差为0.1
        self.frame_num = int(sz / self.len)
        self.frames = np.zeros((self.frame_num, self.len))
        for i in range(self.frame_num):
            self.frames[i, :] = data[i * self.len : (i + 1) * self.len]
        print("Framing process completed.")

    # deprecated?
    def saveAudio(self, path:str):
        pass

    def drawResult(self, path, draw_filter = False):
        plt.figure(1)
        plt.plot(np.arange(self.origin_data.size), self.origin_data, c = "k")
        plt.title("Original data (Down-sampled)")
        plt.figure(2)
        plt.plot(np.arange(self.frames.size), self.frames.ravel(), c = "k")
        plt.title("Noised data")
        if draw_filter == True:
            plt.figure(3)
            plt.plot(np.arange(self.data_size), self.output, c = "k")
            plt.title("Non-white-noised data (Kalman-Filtered)")
        plt.show()
        np.savetxt(path, self.output)

    # 得到状态转移矩阵A
    @staticmethod
    def getTransition(coeffs, order):
        _A = np.zeros((order, order))
        _A[-1, :] = coeffs
        _A[:-1, 1:] = np.eye(order - 1)
        return _A

    """
        滤波主函数
        ------------
        Parameters
        max_iter LPC与KF交替迭代次数
    """
    def filtering(self, max_iter = 3):
        # 对每一帧进行操作
        start_time = time.time()
        for k in range(self.frame_num):
            print("Processing frame: %d / %d"%(k + 1, self.frame_num))
            self.oldState = self.statePost.copy()
            if k == 0:
                start_pos = self.ord
                self.statePost = self.frames[0, :self.ord].reshape(-1, 1)     # 初始后验估计
            else:
                start_pos = 0
            
            for i in range(max_iter):       # 每一帧KF与LPC交替迭代 max_iter次
                # 此处是LPC操作 coeffs 为系数向量
                
                coeffs, self.stateNoiseErrorCov = KalmanAudio.LPC(self.output[k * self.len : (k + 1) * self.len], self.ord)
                coeffs = coeffs[1:]         # librosa LPC操作第一位应该是x(n), 降序 x(n), x(n-1)...
                coeffs = coeffs[::-1]
                A = KalmanAudio.getTransition(coeffs, self.ord)
                for n in range(start_pos, self.len):    # 每一帧的所有数据进行滤波 当为第一帧时，第一次滤波从order索引处开始
                    # ================ KF 操作开始 ===================
                    self.statePre[:-1] = self.statePost[1:]
                    self.statePre[-1] = coeffs @ self.statePost
                    self.statePre = A @ self.statePost
                    self.stateCovPre = (A @ self.stateCovPost @ A.transpose()
                        +  self.stateNoiseErrorCov * self.NC @ self.NC.transpose()
                    )
                    K = self.stateCovPre[:, -1].reshape(-1, 1) / (self.stateCovPre[-1, -1] + self.measureNoiseCov)
                    self.statePost = self.statePre + K * (self.frames[k, n] - self.statePre[-1]) 
                    self.stateCovPost = (np.eye(self.ord) - K @ self.H) @ self.stateCovPre
                    #=================== KF 结束 ======================

                    insert_pos = k * self.len + self.ord + n - start_pos + 1
                    if insert_pos >= self.data_size:
                        break
                    self.output[insert_pos - 1] = self.statePost[0]
                start_pos = 0
                if i < max_iter - 1:
                    self.statePost = self.oldState.copy()      # 保持初始后验估计
        end_time = time.time()
        print("Running time: ", end_time - start_time)

# FIR 滤波器
def showFFT(data, show_phase = False):
    res = np.fft.fft(data)
    amps = np.sqrt(res * res.conjugate())
    phases = np.arctan(res.imag / (res.real + np.ones_like(res) * 1e-7))
    size = res.size
    start_pos = int(size / 2) - size
    plt.figure(1)
    plt.plot(np.arange(start_pos, start_pos + size), amps, c = 'k')
    plt.title("Amplitude")
    if show_phase:
        plt.figure(2)
        plt.plot(np.arange(start_pos, start_pos + size), phases, c = 'k')
        plt.title("Phases")
    plt.show()

if __name__ == "__main__":
    number = 2
    inpath = path_prefix + str(number) + ".wav"
    outpath = "..\\data\\" + str(number) + ".csv"
    ka = KalmanAudio(inpath)
    print("Start filtering...")
    ka.filtering()
    print("Filter process completed.")
    ka.drawResult(outpath, True)
    # inpath = "..\\full\\%d\\1.wav"%(number)
    # data, sr = lr.load(inpath)
    # showFFT(data)

    