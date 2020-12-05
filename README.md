# AudioDSP

---
## 语音DSP实验项目

![]( https://visitor-badge.glitch.me/badge?page_id=<[your_page_id](https://github.com/Enigmatisms/AudioDSP)>)

---

#### I. 项目说明

| Python库     | 官方链接🔗                                 |
| ------------ | ----------------------------------------- |
| Scikit-learn | https://scikit-learn.org/stable/          |
| Numpy        | https://numpy.org                         |
| Matplotlib   | https://matplotlib.org                    |
| LibROSA      | https://librosa.org/doc/latest/index.html |
| Keras        | https://github.com/keras-team             |
| Cython       | https://cython.org                        |

其中，Keras与Cython环境的配置相对麻烦。

- Keras需要CUDA / cuDNN / Tensorflow支持，可以选择绕开Anaconda

- Cython 在amd64平台上运行时至少需要VS2017的某个dll支持，并且需要配置mingw64编译器，才能对.pyx文件进行预编译
  - 没有Cython支持很可能运行不了VAD.py文件！

其余库均可pip

---

#### II. 工程文件内容

| 文件夹(点击导航)                                             | 作用                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [DTW_mi](https://github.com/Enigmatisms/AudioDSP/tree/master/DTW_mi) | [83米唯实](https://github.com/hhhwmws0117)实现的DTW          |
| [Speaker_recognition](https://github.com/Enigmatisms/AudioDSP/tree/master/Speaker_recognition) | [83米唯实](https://github.com/hhhwmws0117)实现的GMM-MFCC模型说话人识别 |
| [matlab](https://github.com/Enigmatisms/AudioDSP/tree/master/matlab/VAD) | [84张颖](https://github.com/ZY-de)实现的VAD                  |
| [neural_network](https://github.com/Enigmatisms/AudioDSP/tree/master/neural%20network) | [84杨恺](https://github.com/yk7333)实现的MFCC神经网络        |
| [py](https://github.com/Enigmatisms/AudioDSP/tree/master/py) | [84何千越](https://github.com/Enigmatisms)VAD优化/Cython/时域法/DTW + 84杨恺实现的FFT |
| [segment](https://github.com/Enigmatisms/AudioDSP/tree/master/segment) | 杂项                                                         |
| [weekly](https://github.com/Enigmatisms/AudioDSP/tree/master/weekly) | 周报与实验报告                                               |

**<u>开源说明</u>**：不要到处耍（我们代码能力有限）。遵循[**<u>【MIT License】</u>**](https://github.com/Enigmatisms/AudioDSP/blob/master/LICENSE)

> Enigmatisms/AudioDSP is licensed under the
>
> ### MIT License
>
> A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.
>
> #### Permissions												Limitations
>
> :heavy_check_mark: Commercial use										:x: Liability
>
> :heavy_check_mark:  Modification											:x: Warranty
>
> :heavy_check_mark:  Distribution
>
> :heavy_check_mark:  Private use

- **<u>感谢参与本项目的小组成员:cocktail::coffee::wine_glass:</u>**

---

#### III. 代码运行

文件夹【py】中含有timeDomain.py（时域法），需要更改

```
path = "..\\segment\\random%d.wav"%(num)
```

为所需路径

【py】中包含的频域法 freqDomain.py类似，但由于这些整合模块都包括了我们已经训练好的模型，或是一些只保存在本地，没有发送到云端的音频/数据集，加载时必定会存在麻烦。如果实在想运行代码，请联系库拥有者。

---

#### IV. 项目进度

- 2020.10.4
  - 测试librosa等库
- 2020.10.5
  - 尝试实现LPC与KF融合的降噪处理
- 2020.10.6
  - KF & LPC模块基本完成
- 2020.10.8
  - KF 模块完成，就是非常的慢 采样率降到6000Hz，71帧每帧长1500，需要LPC KF交替迭代三次 总时间9.8s才能跑完
- 2020.10.10
  - KF 模块可能无法继续进行速度优化了吧，采样率4000Hz，7s内处理完17s的音频，不够块，但是效果不错
  - KF 已经得到了验证，输出结果效果（除了1以外）都不错，没有明显失真
  - 特征提取算法写好，准备放入分类器进行尝试
- 2020.10.15
  - 人工分割数据集已经建立，中规模数据集（每个数字大概90个数据）分类正确率78%（时域）
- 2020.10.20
  - 张颖的分割代码转为Python，分割效果还需要进一步优化
- 2020.11
  - 完成了VAD优化，Cython加速
  - 完成了MFCC网络构建，数字识别率85%
- 2020.12
  - 完成了模块整合
  - 完成了DTW（准确率61%，6模板数字）
  - 完成了实测