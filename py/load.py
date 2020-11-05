import librosa
import numpy as np
import librosa as lr
from time import time
import matplotlib.pyplot as plt
import sys
import audioop
import wave
import audioread

def differentLoad(path):
    start_time = time()
    data, sr = lr.core.load(path)
    med_time = time()
    f = wave.open(path,'rb')
    params = f.getparams()
    nchans, sample_w, framerate, nframes = params[:4]
    strData = f.readframes(nframes)
    strData, _ = audioop.ratecv(strData, 2, nchans, framerate, 22050, None)
    waveData = np.array(np.frombuffer(strData, dtype=np.int16), dtype = np.float32)
    # waveData = librosa.core.resample(waveData, framerate, sr)
    waveData /= 36000
    # for i, buf in enumerate(f):
    #     print(i, buf)
    # with audioread.audio_open(path) as f:
    #     print(f.channels, f.samplerate, f.duration)
    #     strData, _ = audioop.ratecv(strData, 2, nchans, framerate, 22050, None)
    #     for buf_i, buf in enumerate(f):
    #         if buf_i < 100:
    #             continue
    #         data = np.frombuffer(buf, np.int16)
    #         plt.plot(range(buf_i*len(data)//2, (buf_i+1)*len(data)//2), data[0::2])
    #         plt.plot(range(buf_i*len(data)//2, (buf_i+1)*len(data)//2), data[1::2])
    #         if buf_i > 120:
    #             break
    end_time = time()
    max_wave = max(waveData)
    max_lbr = max(data)
    print("Wave rate:%d and librosa rate: %d"%(framerate, sr))
    print("Wave / librosa: ", max_wave / max_lbr)
    print("Librosa time: %f, wave time: %f\n"%(med_time - start_time, end_time - med_time))
    plt.plot(np.arange(waveData.size), waveData, c = 'b')
    plt.figure(2)
    plt.plot(np.arange(data.size), data, c = 'r')
    plt.show()
    return max_wave / max_lbr

if __name__ == "__main__":
    number = int(sys.argv[1])
    mean = 0
    for i in range(2, 3):
        path = "..\\segment\\%d\\%d0%d.wav"%(number, number, i)
        mean += differentLoad(path)
    print("Mean value: ", mean / 8)