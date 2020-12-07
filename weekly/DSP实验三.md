<center><font size = 6 color='black'>DSP DTW<font face='新宋体'><b>数字识别实验报告</b></font></font></center>

---

<center><font size = 2>小组成员：何千越 *，杨恺*，米唯实*，张颖，安若鼎</center></font>

<center><font size = 2>（标有星号的成员参与了对应任务，未标星号的成员参与实验其他部分）</center></font>

**摘要** 在实验三中，我们通过使用实验一的优化双门限VAD算法得到需要匹配的语音，每个数字随机选择6个模板后，对测试集中每一个数字进行匹配。匹配使用了DTW算法，先是对所有的语音进行分帧加窗，对每个短时分析端进行MFCC系数求解，并将其重构为一个行向量，最后得到MFCC二维矩阵。对每一行进行欧式距离求解，使用DP思想的DTW算法进行cost计算。DTW数字识别的准确度为61%（284个测试数字）。

**关键词：**DTW算法，MFCC系数

---

<font size = 5> **1   引 言 **</font>

​		常见的语音识别方法有动态时间归整技术(DTW)、矢量量化技术(VQ)、隐马尔可夫模型(HMM)、基于段长分布的非齐次隐马尔可夫模型和人工神经元网络(ANN)。DTW是较早的一种模式匹配和模型训练技术，它应用动态规划的思想成功解决了语音信号特征参数序列比较时时长不等的难题，在孤立词语音识别中获得了良好性能。虽然HMM模型和ANN在连续语音大词汇量语音识别系统优于DTW，但由于DTW算法计算量较少、无需前期的长期训练，也很容易将DTW算法移植到单片机、DSP上实现语音识别且能满足实时性要求，故其在孤立词语音识别系统中仍然得到了广泛的应用。实验三将针对孤立语音字识别问题，构建DTW算法进行问题的解决。

<font size = 5> **2  算 法 实 现 综 述**</font>

##### 2.1 DTW语音数字识别

​		DTW由于可以实现非线性地对齐模板以及待匹配的目标，在语音孤立词识别的过程中，可以方便我们进行时域上的非线性映射。由于每个数字在进行发声时，每次发声的时间可能不一样，甚至每个帧对应的情况都不一样。那么直接的匹配 / 线性的匹配显然是不可行的**<u>[[1]](#paper)</u>**。为此，为了减少计算量以及进行更加鲁棒的DTW识别，我们设计了如如下图所示的DTW语音数字识别算法：

![](C:\Users\15300\Desktop\图片1.png)

<center><font size = 1>图1. DTW数字识别算法流程</font></center>

​		通过以上算法，可以得到最后的DTW算法结果。

​		在第三节中，我们会对如上提到的实验三 / 实验四算法的原理以及相关实现进行更加细致的说明。**<u>3.1</u>** 中将介绍MFCC求解的原理以及算法求解的过程。**<u>3.2</u>**中将会介绍DTW的预处理 / 基本实现。

​		在第四部分中，我们将对 DTW实验结果进行说明。在第五部分中我们将给出关于两个实验的结论。

<font size = 5> **3  原 理 及 实 现**</font>

##### 3.1 MFCC的求解过程

​		**<u>分帧：</u>**对于端点检测部分，在实验一中我们已经使用传统优化的VAD双门限算法，得到了比较好的端点分割结果。那么对分割结果（目标音频），由于其时常过长，在时域上进行DTW显然是不可取的，而在频域上，如果对于一段音频直接进行MFCC系数求解，那我们将会失去时间域上的伸缩映射能力，那么整个算法就会退化成一个**最邻近问题**。指的是：我们只根据每个音频求取一个MFCC系数矩阵，使用这一个MFCC系数矩阵，找到与目标距离最近（欧式距离）的模板。那么整个算法实际上可以用一个KNN来进行表述，但这没有什么很大的意义。

​		我们需要对语音进行分帧处理，每一帧进行MFCC，当分帧足够细致时，我们可以根据帧与帧之间的MFCC欧式距离的关系，得到模板与目标的每一帧之间的对应关系。这就实现了DTW中的非线性映射，当然这会对应时域上的伸缩。在本问题中，考虑到数字的长度（在采样频率为22050Hz时）一半会有8000-11000个数据点的采样分割结果，我们取每一帧长度为400，帧移为长度的0.9倍。

​		**<u>加窗：</u>**由于MFCC实际是一个频域特征，为了使经过DSP系统的信息失真能够减小（矩形窗特性），我们可以在频域上进行加窗。而实际上，由于加窗环节，模板与目标可以同时选择加窗或者不加窗，最终对实验结果的影响并不大。

​		**<u>频域变换</u>**：首先进行FFT。适当选取FFT序列长度，以提高FFT对频域信号的计算精度。通过FFT之后，可以得到复频域特征。我们对复频域FFT计算值取幅值的平方，将它转换为频域上的能量分布；将能量谱通过一组Mel尺度的三角形滤波器组，对频谱进行平滑化，并消除谐波的作用，突显原先语音的共振峰；计算每个滤波器输出的对数能量，经离散余弦变换（DCT）得到MFCC系数；然后计算对数能量；最后提取动态差分参数（包括一阶差分和二阶差分等）。整个MFCC求解过程如下：

![](C:\Users\15300\Desktop\图片2.png)

<center><font size = 1>图2. MFCC数字识别算法流程</font></center>

##### 3.2 DTW的预处理与实现

​		分帧结束后，reference语音的帧数为M，而target语音的帧数为N。如果要将两者进行匹配，需要计算一个MFCC每一帧的cost。使用欧式距离表示两个MFCC系数个数相等的向量的差距。显然，当我们匹配到target语音（记为$V_t$）的第n帧时（对应reference（记为$V_r$）第m帧），对下一次的匹配，我们有如下三种选择**<u>[[2]](#paper)</u>**：

- $V_t$第n+1帧匹配$V_r$第m帧（此帧附近，$V_t$发音时间更长）
- $V_t$第n帧匹配$V_r$第m+1帧（此帧附近，$V_t$发音时间更短）
- $V_t$第n+1帧匹配$V_r$第m+1帧（此帧附近，$V_t$与$V_r$发音时间类似）

​		那么根据动态规划的最优性原则，我们需要选择最小的cost策略，则移动到这三个位置中对应可以使最终cost最小的位置上。DTW算法的流程如下图所示：

![](C:\Users\15300\Desktop\图片3.png)

<center><font size = 1>图3. DTW Cost计算算法流程</font></center>

<font size = 5> **4 实 验 结 果**</font>

##### 4.1 DTW实验结果

​		一下分别为几个不同的模板（0，4，5，8）与数字0进行识别的cost移动匹配策略图。下图展示的矩阵并不是cost矩阵本身，而是累计误差矩阵（accumulated cost），每一个块对应的都是到达本块策略下的最小cost值。

![](C:\Users\15300\Desktop\00.png)

<center><font size = 1>图4. 0-0模板匹配</font></center>

<img src="C:\Users\15300\Desktop\4.png" style="zoom: 40%;" />

<center><font size = 1>图5. 0-4模板匹配</font></center>

<img src="C:\Users\15300\Desktop\55.png" style="zoom:40%;" />

<center><font size = 1>图6. 0-5模板匹配</font></center>

<img src="C:\Users\15300\Desktop\8.png" style="zoom: 50%;" />

<center><font size = 1>图7. 0-8模板匹配</font></center>

​		可以看出，0与8容易混淆（在频域上的反映可能比较类似）。但是，在模板较多的情况下，仍然能保证61%的准确率。

​		我们从实验一中得到VAD分割结果中，每个数字随机抽选了6个作为模板音频。测试集仍然是实验一、二中的测试集（每个数字约27个），最后的程序输出如下：

```python
# 部分计算结果图，最后在284个数字中，108个识别错误，正确率为61.97%
Number proba:  [8811.71750565 8462.00619319 8170.00928084 7777.32479082 8213.71679304
 6665.00623055 6791.8801597  8994.19987872 7415.17324755 7054.23290575]
Process: number 9, NO.24
Number proba:  [7264.21346735 8666.96314182 7674.3397296  7320.55736616 7690.61092088
 6742.26061425 6137.83335161 8954.83465703 7442.96182394 6446.1910363 ]
Process: number 9, NO.25
Number proba:  [8268.43429797 8949.1343854  8922.02208651 8313.33402386 8203.87589562
 6512.59605083 6799.93163519 9043.07538964 8817.87900312 6882.26768787]
Process: number 9, NO.26
Number proba:  [6651.98815094 7709.99658858 7717.06869234 7506.527778   7502.09236453
 6242.65696491 5849.29438576 7823.93085896 7454.65305762 6216.27378433]
Process: number 9, NO.27
# 以上是正在处理数字9时的输出
Process completed.
Total 108 / 284 Faults. Ratio for correct recogniton: 0.619718
```

<font size = 5> **5  结 论**</font>

​		对于实验三，基于DTW的数字识别，我们发现，DTW的实现较为简单，并且有非常直观的可解释性。相比起分类器而言，其原理更加简单，但是在计算时间上会相对较长。分类器学习结束之后，每一次predict操作几乎不耗费时间，而DTW除了需要计算MFCC之外，还需要对每一个MFCC矩阵的行进行欧式距离的计算。本实验中使用到了scipy库中的cdist函数，对DTW进行了加速，才达到一个较快的效果，而普通DTW需要耗费1-3秒才能得到一个数字预测结果（也取决于模板分帧 / 目标分帧帧数）。最后我们的DTW数字识别精度达到了61%。

<font size = 5> **6  参 考 文 献**</font>

<span id ='paper'>

[1] 张军、李学斌，一种基于DTW的孤立词语音识别算法，计算机仿真，2009, 26(10)

[2] 冯晓亮、于水源，三种基于DTW的模板训练方法的比较，中国传媒大学声学研究所

</span>

<font size = 5> **7  附 录**</font>

​		也可以直接转到[【链接🔗：Github:AudioDSP Repository】](https://github.com/Enigmatisms/AudioDSP)查看相关代码。

​		Python代码：

1. DTW算法实现：

```python
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
```



