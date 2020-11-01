cimport numpy as np
cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def annealing(np.ndarray[np.float64_t, ndim=1] seg, int end, int max_iter = 30, float ab_thresh = 0.03, float start_temp = 0.3):
    cdef int now_pos = end
    cdef int seg_len = len(seg)
    cdef int min_pos = now_pos
    cdef int tipping_cnt = 0
    cdef int i = 0
    cdef int step = 0
    cdef int tmp_pos = now_pos
    cdef float this_val = 0.0
    cdef float min_val = seg[now_pos]
    for i in range(max_iter):
        if i > 6:
            step = np.random.choice((1, 2)) * np.random.choice((-1, 1))
        elif i > 2:
            step = np.random.choice((1, 2, 3)) * np.random.choice((-1, 1))
        else:
            step = np.random.choice((1, 1, 2, 2, 3))
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
            if  np.random.uniform(0, 1) < np.exp(- (now_val - min_val) / (start_temp / float(i + 1))):
                now_pos = tmp_pos
        this_val = seg[now_pos]
        if this_val < ab_thresh:
            return min_pos
        elif this_val < 2 * ab_thresh:
            if np.random.uniform(0, 1) < np.exp( - tipping_cnt / 2):
                return min_pos
            tipping_cnt += 1
    return min_pos

@cython.boundscheck(False)
@cython.wraparound(False)
def calculateMeanAmp(np.ndarray[np.float64_t, ndim=2] y):
    cdef float wlen = float(y.shape[0])
    cdef int fn = y.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] amps = np.zeros(fn)
    cdef int i = 0
    for i in range(fn):
        amps[i] = sum(abs(y[:, i])) / wlen
    return amps

@cython.boundscheck(False)
@cython.wraparound(False)
def faultsFiltering(np.ndarray[np.float64_t, ndim=1] amps, starts, ends, float thresh):
    cdef int max_val = max(amps)
    _starts = []
    _ends = []
    if thresh > max_val / 4:
        thresh = max_val / 4
    for start, end in zip(starts, ends):
        if np.mean(amps[start:end]) > thresh and end - start >= 12: # 单门限阈值 长度限制(__min_ken__)
            _starts.append(start)
            _ends.append(end)
    return _starts, _ends

@cython.boundscheck(False)
@cython.wraparound(False)
def zeroCrossingRate(np.ndarray[np.float64_t, ndim = 2] y, int fn):
    cdef int i = 0
    cdef int j = 0
    cdef int wlen = y.shape[0] - 1
    cdef np.ndarray[np.float64_t, ndim=1] zcr = np.zeros(fn, dtype = np.float64)
    for i in range(fn):
        for j in range(wlen):
            if (y[j, i] > 0 and y[j + 1, i] < 0) or (y[j, i] < 0 and y[j + 1, i] > 0):
                zcr[i] += 1
    return zcr

@cython.boundscheck(False)
@cython.wraparound(False)
def findSegment(np.ndarray[np.int32_t, ndim=1] expr):
    cdef int i = 0
    cdef np.ndarray[np.int32_t, ndim=1] voice_index
    cdef int size = 0
    if expr[0] == 0:
        voice_index = np.arange(expr.size)[expr == 1]
    else:
        voice_index = expr
    size = voice_index.size
    starts = [voice_index[0]]
    ends = []
    for i in range(voice_index.size - 1):
        if voice_index[i + 1] - voice_index[i] > 1:
            starts.append(voice_index[i + 1])
            ends.append(voice_index[i])
    ends.append(voice_index[size - 1])
    return starts, ends

@cython.boundscheck(False)
@cython.wraparound(False)
def enframe(np.ndarray[float, ndim = 1] x, int win, int inc = 400):
    cdef int nx = len(x)
    cdef int nf = int(np.round((nx - win + inc) / inc))
    cdef int i = 0
    cdef np.ndarray[np.float64_t, ndim=2] f = np.zeros((win, nf), dtype = np.float64)
    for i in range(nf):
        temp = x[inc * i:inc * i + win]
        f[:temp.size, i] = x[inc * i:inc * i + win].ravel()      # 没有帧移
    return f
