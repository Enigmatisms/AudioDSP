%��ȡ�����ļ�
[x,Fs]=audioread('C:\Users\zhang\Desktop\����ʾ��\8.wav');%x��ʾ��FsΪ�����ʼ�¼��n�������źŲ���������
time=(0:length(x)-1)/Fs;% ������źŵ�ʱ��̶�
xmax=max(abs(x));
x=x/xmax;%��һ��

%��֡�Ӵ�����ö�ʱ����,��ʱƽ��������
wlen=1500;
inc=700;%����֡����֡��(���ݲ�����ȷ������֤ÿ֡ʱ����30ms���ң�֡��Ϊ֡����0~1/2��
w=hanning(wlen);%ѡ�������Ӵ�
X=enframe(x,w,inc)';%��֡��ÿ�д���һ֡
fn=size(X,2);%֡��
En=zeros(1,fn);
for i=1 : fn
    u=X(:,i);              % ȡ��һ֡
    u2=u.*u;               % �������
    En(i)=sum(u2);         % ��һ֡�ۼ����
end
=zc2(X,fn);
subplot 311; plot(time,x,'k'); % ����ʱ�䲨�� 
title('��������');
ylabel('��ֵ'); xlabel(['ʱ��/s' 10 '(a)']);
frameTime=frame2time(fn,wlen,inc,Fs);   % ���ÿ֡��Ӧ��ʱ��
subplot 312; plot(frameTime,En,'k')     % ������ʱ����ͼ
title('��ʱ����');
ylabel('��ֵ'); xlabel(['ʱ��/s' 10 '(b)']);
subplot 313; plot(frameTime,zcr,'k'); grid;%������ʱ������ͼ
title('��ʱƽ��������');
ylabel('��ֵ'); xlabel(['ʱ��/s' 10 '(c)']);







