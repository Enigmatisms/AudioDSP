disp('正在计算参考模板的参数...')
for i=1:9   %改变需要修改
	filename = sprintf('%da.wav',i);
	x=filename;
    [x,fs]=wavread(x);
	[x1 x2] = vad(x);
	mfcc_feature = MFCC(x);
	mfcc_feature = mfcc_feature(x1-2:x2-2,:);
	reference(i).mfcc = mfcc_feature;
end

disp('正在计算测试模板的参数...')
for i=1:2   %改变需要修改
    filename = sprintf('%db.wav',i);
	x=filename;
    [x,fs]=wavread(x);    
	[x1 x2] = vad(x);
	mfcc_feature = MFCC(x);
	mfcc_feature = mfcc_feature(x1-2:x2-2,:);
	test(i).mfcc = mfcc_feature;
end
disp('正在进行模板匹配...')
distance = zeros(2,9);%初始化%改变需要修改
for i=1:2   %改变需要修改
    for j=1:9   %改变需要修改
	distance(i,j) = dtw(test(i).mfcc, reference(j).mfcc);
    end
end
disp('正在计算匹配结果...')
for i=1:2   %改变需要修改
	[d,j] = min(distance(i,:));
	fprintf('测试模板 %d 的识别结果为：%d\n', i, j);
end
