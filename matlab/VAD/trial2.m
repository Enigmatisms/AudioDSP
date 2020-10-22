number=9;
filename1=strcat('?D:\AudioDSP\voice_stream\',num2str(number),'.wav');
[x,fs]=audioread("D:\AudioDSP\voice_stream\5.wav");  % 读入数据文件
x=x/max(abs(x));    % 幅度归一化
N=length(x);                            % 取信号长度
time=(0:N-1)/fs;                        % 计算时间
pos = get(gcf,'Position');              % 作图
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-200)]);
plot(time,x,'k');         
title(strcat('语音文件',int2str(number),'.wav的端点检测'));
ylabel('幅值'); axis([0 max(time) -1 1]); grid;
xlabel('时间/s');
wlen=1500; inc=700;                       % 分帧参数
IS=0.5; overlap=wlen-inc;               % 设置IS(前置无话段的长度）
NIS=fix((IS*fs-wlen)/inc +1);           % 计算NIS（前导无话段的帧数）
fn=fix((N-wlen)/inc)+1;                 % 求总帧数
frameTime=frame2time(fn, wlen, inc, fs);% 计算每帧对应的时间
[voiceseg,vsl,SF,NF]=vad2(x,wlen,inc,NIS);  % 端点检测
for k=1 : vsl                           % 画出起止点位置
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    nx3=voiceseg(k).duration;
    fprintf('%4d   %4d   %4d   %4d\n',k,nx1,nx2,nx3);
    line([frameTime(nx1) frameTime(nx1)],[-1.5 1.5],'color','r','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1.5 1.5],'color','b','LineStyle','-');
end
