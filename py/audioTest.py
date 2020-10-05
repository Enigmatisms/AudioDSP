#-*-coding:utf-8-*-

import librosa as lr
import pyAudioAnalysis
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    aud_path = ".\\raw\\1.wav"
    data, sample_rate = lr.load(aud_path)
    xs = np.arange(len(data))


    print("Sampling rate:", sample_rate)
    plt.plot(xs, data, c = "k")

    # =============== 简单尝试了一下 SFTF 特征=============
    # feat1 = lr.feature.chroma_stft(data)
    # print("Feature stft:", feat1.shape, " type:", feat1.dtype)
    # feat1 = (feat1 * 255).astype(np.uint8)
    # plt.figure(2)
    # plt.imshow(feat1)

    resamp = lr.resample(data, sample_rate, 8000)
    plt.figure(2)
    plt.plot(np.arange(len(resamp)), resamp, c = "k")
    plt.scatter(np.arange(len(resamp)), resamp, c = "k", s = 2)

    plt.show()

