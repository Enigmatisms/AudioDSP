number=9;
filename1=strcat('?D:\AudioDSP\voice_stream\',num2str(number),'.wav');
[x,fs]=audioread("D:\AudioDSP\voice_stream\5.wav");  % ���������ļ�
x=x/max(abs(x));    % ���ȹ�һ��
N=length(x);                            % ȡ�źų���
time=(0:N-1)/fs;                        % ����ʱ��
pos = get(gcf,'Position');              % ��ͼ
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)-200)]);
plot(time,x,'k');         
title(strcat('�����ļ�',int2str(number),'.wav�Ķ˵���'));
ylabel('��ֵ'); axis([0 max(time) -1 1]); grid;
xlabel('ʱ��/s');
wlen=1500; inc=700;                       % ��֡����
IS=0.5; overlap=wlen-inc;               % ����IS(ǰ���޻��εĳ��ȣ�
NIS=fix((IS*fs-wlen)/inc +1);           % ����NIS��ǰ���޻��ε�֡����
fn=fix((N-wlen)/inc)+1;                 % ����֡��
frameTime=frame2time(fn, wlen, inc, fs);% ����ÿ֡��Ӧ��ʱ��
[voiceseg,vsl,SF,NF]=vad2(x,wlen,inc,NIS);  % �˵���
for k=1 : vsl                           % ������ֹ��λ��
    nx1=voiceseg(k).begin; nx2=voiceseg(k).end;
    nx3=voiceseg(k).duration;
    fprintf('%4d   %4d   %4d   %4d\n',k,nx1,nx2,nx3);
    line([frameTime(nx1) frameTime(nx1)],[-1.5 1.5],'color','r','LineStyle','-');
    line([frameTime(nx2) frameTime(nx2)],[-1.5 1.5],'color','b','LineStyle','-');
end
