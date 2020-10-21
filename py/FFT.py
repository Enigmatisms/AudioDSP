'''
yk 2020-10-21
FFT算法的实现（基于频率抽取的算法）
修正了算法的一些错误
'''


import numpy as np

def butterfly(x1,x2,w):
    return (x1+x2,(x1-x2)*w)
def Wn(N,n):
    return  np.exp(-1j*2*np.pi*n/N) 
#求出X(k)对应的最后一层蝴蝶运算结果x(p(k))
#N=8时 X(0)-->x(0) X(4)-->x(1) X(1)-->x(4)
def P(x,N):
    length=int(np.log2(N))
    b=np.zeros(length)
    p=0
    #转化为二进制数并且倒置
    for i in range(length):
        b[i]=x%2
        x=x//2
    for i in range(length):
        p+=b[length-i-1]*(2**i)
    return int(p)
#FFT算法 N为采样点数
def FFT(x,N):
    length=len(x)
    x_f=np.zeros(N-length)
    x=np.hstack((x,x_f))
    depth=int(np.log2(N))
    x_r=x
    x_i=np.zeros(N)
    x1_r=np.zeros(N)                     #_r实部，_i虚部
    x1_i=np.zeros(N)
    #基于频率抽取的FFT算法
    for j in range(depth):
        j+=1
        F=np.zeros(N)
        if j%2==1:             #采用了x1,x循环使用的方式
            for i in range(N):
                if F[i]==0:    #x[i],x[i+N/2^j]进行蝴蝶运算 分别求得实部和虚部值
                    x1_r[i]=np.real(butterfly(x_r[i]+1j*x_i[i],x_r[i+int(N/2**j)]+1j*x_i[i+int(N/2**j)],Wn(int(N/2**(j-1)),i%int(N/2**j)))[0])
                    x1_i[i]=np.imag(butterfly(x_r[i]+1j*x_i[i],x_r[i+int(N/2**j)]+1j*x_i[i+int(N/2**j)],Wn(int(N/2**(j-1)),i%int(N/2**j)))[0])
                    x1_r[i+int(N/2**j)]=np.real(butterfly(x_r[i]+1j*x_i[i],x_r[i+int(N/2**j)]+1j*x_i[i+int(N/2**j)],Wn(int(N/2**(j-1)),i%int(N/2**j)))[1])
                    x1_i[i+int(N/2**j)]=np.imag(butterfly(x_r[i]+1j*x_i[i],x_r[i+int(N/2**j)]+1j*x_i[i+int(N/2**j)],Wn(int(N/2**(j-1)),i%int(N/2**j)))[1])
                    F[i]=F[i+int(N/2**j)]=1; 
        else:
            for i in range(N):
                if F[i]==0:
                    x_r[i]=np.real(butterfly(x1_r[i]+1j*x1_i[i],x1_r[i+int(N/2**j)]+1j*x1_i[i+int(N/2**j)],Wn(int(N/2**(j-1)),i%int(N/2**j)))[0])
                    x_i[i]=np.imag(butterfly(x1_r[i]+1j*x1_i[i],x1_r[i+int(N/2**j)]+1j*x1_i[i+int(N/2**j)],Wn(int(N/2**(j-1)),i%int(N/2**j)))[0])
                    x_r[i+int(N/2**j)]=np.real(butterfly(x1_r[i]+1j*x1_i[i],x1_r[i+int(N/2**j)]+1j*x1_i[i+int(N/2**j)],Wn(int(N/2**(j-1)),i%int(N/2**j)))[1])
                    x_i[i+int(N/2**j)]=np.imag(butterfly(x1_r[i]+1j*x1_i[i],x1_r[i+int(N/2**j)]+1j*x1_i[i+int(N/2**j)],Wn(int(N/2**(j-1)),i%int(N/2**j)))[1])
                    F[i]=F[i+int(N/2**j)]=1; 
    if depth%2==1:
        for i in range(N):
                x_r[i]=x1_r[P(i,N)]
                x_i[i]=x1_i[P(i,N)]
        return x_r,x_i
    else:
        for i in range(N):
                x1_r[i]=x_r[P(i,N)]
                x1_i[i]=x_i[P(i,N)]
        return x1_r,x1_i
if __name__=='__main__':
    re,im=FFT(np.array([1,1,1,1],dtype=np.float32),8)
    print("real part of WN(k) is: ",re)     
    print("imaginary part of WN(k) is: ",im)