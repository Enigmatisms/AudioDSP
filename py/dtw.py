'''
    yk  2020-10-23
    DTW函数
    输入为两个mfcc向量(shape值可以不一样)，输出为差异值
    用每一个语音信号提取的mfcc特征跑dtw效果不是很好（不同数字和相同数字之间相差20%）
    估计要每一帧提取mfcc，再进行dtw计算
'''

def dtw(mfcc1, mfcc2):
    M, N = len(mfcc1), len(mfcc2)
    #求两个数之间差异
    d=lambda x,y: abs(x-y)
    #初始化矩阵
    cost = np.ones((M, N))
    cost[0, 0] =d(mfcc1[0], mfcc2[0])
    #计算每一点的cost值
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(mfcc1[i], mfcc2[0])
    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(mfcc1[0], mfcc2[j])

    # 更新权值，选择最小的cost值
    for i in range(1, M):
        for j in range(1, N):
            choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(mfcc1[i], mfcc2[j])
                   
    return cost[-1, -1]