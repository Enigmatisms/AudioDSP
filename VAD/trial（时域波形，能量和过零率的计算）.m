%读取语音文件
[x,Fs]=audioread('C:\Users\zhang\Desktop\语音示例\8.wav');%x表示以Fs为采样率记录的n个声音信号采样点数组
time=(0:length(x)-1)/Fs;% 计算出信号的时间刻度
xmax=max(abs(x));
x=x/xmax;%归一化

%分帧加窗并求得短时能量,短时平均过零率
wlen=1500;
inc=700;%定义帧长和帧移(根据采样率确定，保证每帧时间在30ms左右，帧移为帧长的0~1/2）
w=hanning(wlen);%选择海宁窗加窗
X=enframe(x,w,inc)';%分帧，每列代表一帧
fn=size(X,2);%帧数
En=zeros(1,fn);
for i=1 : fn
    u=X(:,i);              % 取出一帧
    u2=u.*u;               % 求出能量
    En(i)=sum(u2);         % 对一帧累加求和
end
zcr=zc2(X,fn);
subplot 311; plot(time,x,'k'); % 画出时间波形 
title('语音波形');
ylabel('幅值'); xlabel(['时间/s' 10 '(a)']);
frameTime=frame2time(fn,wlen,inc,Fs);   % 求出每帧对应的时间
subplot 312; plot(frameTime,En,'k')     % 画出短时能量图
title('短时能量');
ylabel('幅值'); xlabel(['时间/s' 10 '(b)']);
subplot 313; plot(frameTime,zcr,'k'); grid;%画出短时过零率图
title('短时平均过零率');
ylabel('幅值'); xlabel(['时间/s' 10 '(c)']);







