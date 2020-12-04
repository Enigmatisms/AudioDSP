function main_test()
load matlab
for i=1:length(test_info)
    [voice_data, Fs] = audioread(strcat(test_info(i).folder, '/', test_info(i).name));
    mfcc_features = get_features(voice_data, Fs);
    
    [d1, log1] = posterior(GMM_models{1}, mfcc_features);
    [d2, log2] = posterior(GMM_models{2}, mfcc_features);
    [d3, log3] = posterior(GMM_models{3}, mfcc_features);
    [d4, log4] = posterior(GMM_models{4}, mfcc_features);
    A = [log1 log2 log3 log4];
    
    fprintf(test_info(i).name);
    choose_people = min(A);
    if choose_people == log1
        fprintf(' label: 1');
        fprintf('\n');
    elseif choose_people == log2
        fprintf(' label: 2');
        fprintf('\n');
    elseif choose_people == log3
        fprintf(' label: 3');
        fprintf('\n');
    elseif choose_people == log4
        fprintf(' label: 4');
        fprintf('\n');
    end
end
end
%%
function [ mfcc_feature ] = get_features( voice_data, fs )
%GET_FEATURES 提取语音信号的MFCC特征
    parameter = 0.92; %预加重系数 0.9 < a < 1。
	voice_data = filter([1-parameter],1,voice_data);%预加重
    mfcc_feature = melcepst(voice_data, fs);	% 提取MFCC特征
end