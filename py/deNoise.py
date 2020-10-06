#-*-coding:utf-8-*-
"""
    声音信号压缩 加噪以及降噪
    高斯白噪声 时域Kalman滤波
"""

import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
from librosa.core import lpc

path_prefix = ".\\raw\\"

"""
    想法是：
        1. 输入一段音频，压缩（采样率8000Hz，尝试一下，原来的采样率为22KHz）加噪（零均值白噪声）
        2. 分帧操作，为了简便，两帧没有重合部分（帧移为1）
        3. 对每一段使用Kalman滤波，模型为AR模型驱动的状态转移方程以及一个定常的观测方程，初始协方差可能需要调参，
            协方差矩阵可能也不是一个对角阵，任意两个采样点之间可能不是独立的
        @todo
        4. 迭代过程中，状态转移的噪声协方差R不是一直固定的（一开始可以建模成完全固定的情况），噪声本身的方差是一定的（高斯白噪声过程决定）
            但是这个方差我们是不知道的，我们需要对其进行估计。LPC由于估计的是系统的最优参数（最小二乘的方法），
            那么必然会存在误差。假设无噪声信号就是系统的输出，那么误差就是加入的零均值噪声
            那么误差的协方差就是LPC的协方差。由于librosa的lpc函数没有协方差输出，所以我们需要把库函数提出来自己加接口。
            但是不知道使用forward还是backward.此情况下R是自适应的
        5. 每次迭代按照 predict（先验：状态转移，使用Xhat(n-1)以及A(n-1) R(n-1)得到先验估计）+ correct（后验：使用当前观测得到观测误差计算Kalman增益）

    难点：
        1. Kalman分帧之后，由于有高阶马尔可夫性，边缘段应该怎么处理？可以舍弃前 k 个点？但我们要求语音信号除了开始部分之后全部需要连续
        2. 大概的想法是：分帧时，由于后验估计是公用的，下一帧可以以这个后验估计开始，这就做到了两帧之间的平滑，后验估计相当于一个滑动窗，在帧内，帧间进行滑动。
            （由于上一帧处理完毕时，后验估计是上一帧最后一个采样点~上一帧最后采样点——阶数处的点）
        3. 一次迭代肯定不够，一次Kalman滤波一帧所有采样点之后求一个线性预测系数，根据这个系数重新预测（可以收敛到弱噪声数据上）
"""
class KalmanAudio:
    def __init__(self, _path, order = 16, length = 400):
        self.ord = order
        self.frames = None          # 二维分帧后的矩阵 分帧后以一行为一帧
        self.frame_num = 0
        self.len = length
        self.loadCompressClip(_path)
        self.NC = np.zeros((order, 1))
        self.H = np.zeros((order, 1))
        self.NC[-1] = 1.0
        self.H[-1] = 1.0

        self.output = np.zeros(self.frames.size)
        self.origin_data = np.zeros(self.frames.size)

        #================ KalmanFilter 模块 ================
        self.oldState = np.zeros((order, 1))             # 历史状态
        self.statePre = np.zeros((order, 1))             # 先验估计
        self.statePost = np.zeros((order, 1))            # 后验估计
        self.stateCovPre = np.eye(order) * 0.75 + np.ones((order, order)) * 0.25            # 状态先验协方差
        self.stateCovPost = np.eye(order) * 0.75 + np.ones((order, order)) * 0.25            # 状态后验协方差
        self.stateNoiseErrorCov = np.zeros((order, 1))   # 状态噪声协方差
        self.measureNoiseCov = np.zeros((order, order))  # 观测协方差

    # librosa函数的提取以及修改（让其可以返回协方差）
    @staticmethod
    def LPC(self, y, sr):
        return self.zeros(self.ord), self.eye(self.ord)


    def loadCompressClip(self, path:str, scale = 0.1):
        _data, sr = lr.load(path)
        data = lr.resample(_data, sr, 8000)
        sz = data.size
        tail = sz % self.len
        data = data[:-tail]
        self.origin_data = data.copy()                      # 保存原数据
        data += np.random.normal(0, scale, data.size)       # 加噪处理，高斯白噪声 标准差为0.1
        self.frame_num = int(sz / self.len)
        self.frames = np.zeros((self.frame_num, self.len))
        for i in range(self.frame_num):
            self.frames[i, :] = data[i * self.len : (i + 1) * self.len]
        print("Framing process completed.")

    # deprecated?
    def saveAudio(self, path:str):
        pass

    # 得到状态转移矩阵A
    @staticmethod
    def getTransition(self, coeffs, order):
        _A = np.zeros((order, order))
        _A[-1, :] = np.flip(coeffs)
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
        for fr in range(self.frame_num):
            self.oldState = self.statePost.copy()
            if fr == 0:
                start_pos = self.ord
                self.statePost = self.frames[fr, :self.ord].reshape(-1, 1)     # 初始后验估计
            else:
                start_pos = 0
            
            for i in range(max_iter):       # 每一帧KF与LPC交替迭代 max_iter次
                # 此处是LPC操作 coeffs 为系数向量
                
                coeffs, self.stateNoiseErrorCov = KalmanAudio.LPC(self.output[fr, :], self.ord)
                coeffs = coeffs[1:]         # librosa LPC操作第一位应该是x(n), 降序 x(n), x(n-1)...
                A = self.getTransition(coeffs, self.ord)
                for p in range(start_pos, self.len):    # 每一帧的所有数据进行滤波 当为第一帧时，第一次滤波从order索引处开始
                    # ================ KF 操作开始 ===================
                    self.statePre = A.dot(self.statePost)
                    self.stateCovPre = (A @ self.stateCovPost @ A.transpose()
                        + self.NC @ self.stateNoiseErrorCov @  self.NC.transpose()
                    )
                    temp = self.H @ self.stateCovPre @ self.H.transpose() + self.measureNoiseCov
                    Minv = np.linalg.solve(temp, np.eye(self.ord))
                    K = self.stateCovPre @ self.H.transpose() @ Minv
                    self.statePost = self.statePre + K * (self.frames[fr, p] - self.statePre[0]) 
                    self.stateCovPost = (np.eye(self.ord) - K @ self.H) @ self.stateCovPre
                    #=================== KF 结束 ======================

                    insert_pos = fr * self.len + self.ord + p - start_pos + 1
                    self.output[insert_pos - self.ord : insert_pos] = self.statePost.transpose()
                start_pos = 0
                if i < max_iter - 1:
                    self.statePost = self.oldState.copy()      # 保持初始后验估计

    