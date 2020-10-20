'''
yk 2020-10-20
FFT算法的实现（基于频率抽取的算法）
跑了几个数，验证了算法是对的~~
'''



import numpy as np

def butterfly(x1,x2,w):
    return (x1+x2,(x1-x2)*w)
def Wn(N,n):
    return  np.exp(-1j*2*np.pi*n/N) 
def FFT(x):
    N=len(x)
    depth=int(np.log2(N))
    x_r=x
    x_i=np.zeros(N)
    x1_r=np.zeros(N)                     #_r实部，_i虚部
    x1_i=np.zeros(N)
    y_r=np.zeros(N)
    y_i=np.zeros(N)
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
                    F[i]=1;F[i+int(N/2**j)]=1; 
        else:
            for i in range(N):
                if F[i]==0:
                    x_r[i]=np.real(butterfly(x1_r[i]+1j*x1_i[i],x1_r[i+int(N/2**j)]+1j*x1_i[i+int(N/2**j)],Wn(int(N/2**(j-1)),i%int(N/2**j)))[0])
                    x_i[i]=np.imag(butterfly(x1_r[i]+1j*x1_i[i],x1_r[i+int(N/2**j)]+1j*x1_i[i+int(N/2**j)],Wn(int(N/2**(j-1)),i%int(N/2**j)))[0])
                    x_r[i+int(N/2**j)]=np.real(butterfly(x1_r[i]+1j*x1_i[i],x1_r[i+int(N/2**j)]+1j*x1_i[i+int(N/2**j)],Wn(int(N/2**(j-1)),i%int(N/2**j)))[1])
                    x_i[i+int(N/2**j)]=np.imag(butterfly(x1_r[i]+1j*x1_i[i],x1_r[i+int(N/2**j)]+1j*x1_i[i+int(N/2**j)],Wn(int(N/2**(j-1)),i%int(N/2**j)))[1])
                    F[i]=1;F[i+int(N/2**j)]=1; 
    if depth%2==0:
        return x_r,x_i
    return x1_r,x1_i
if __name__=='__main__':
    re,im=FFT(np.array([1,2.5,3,4]))
    print("real part of WN(k) is: ",re)     
    print("imaginary part of WN(k) is: ",im)