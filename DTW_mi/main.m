disp('���ڼ���ο�ģ��Ĳ���...')
for i=1:9   %�ı���Ҫ�޸�
	filename = sprintf('%da.wav',i);
	x=filename;
    [x,fs]=wavread(x);
	[x1 x2] = vad(x);
	mfcc_feature = MFCC(x);
	mfcc_feature = mfcc_feature(x1-2:x2-2,:);
	reference(i).mfcc = mfcc_feature;
end

disp('���ڼ������ģ��Ĳ���...')
for i=1:2   %�ı���Ҫ�޸�
    filename = sprintf('%db.wav',i);
	x=filename;
    [x,fs]=wavread(x);    
	[x1 x2] = vad(x);
	mfcc_feature = MFCC(x);
	mfcc_feature = mfcc_feature(x1-2:x2-2,:);
	test(i).mfcc = mfcc_feature;
end
disp('���ڽ���ģ��ƥ��...')
distance = zeros(2,9);%��ʼ��%�ı���Ҫ�޸�
for i=1:2   %�ı���Ҫ�޸�
    for j=1:9   %�ı���Ҫ�޸�
	distance(i,j) = dtw(test(i).mfcc, reference(j).mfcc);
    end
end
disp('���ڼ���ƥ����...')
for i=1:2   %�ı���Ҫ�޸�
	[d,j] = min(distance(i,:));
	fprintf('����ģ�� %d ��ʶ����Ϊ��%d\n', i, j);
end
