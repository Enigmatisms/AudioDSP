GMM_num = 10;
train_path = 'E:\Speaker-Recognition/train'
test_path = 'E:\Speaker-Recognition/test'
train_info = dir(train_path);
speakers_num = length(train_info) - 2;
test_info = dir(strcat(test_path, '/*.wav'));
%
GMM_features = cell(1,speakers_num);
for i=1:speakers_num
    tem_info = dir(strcat(train_info(2 + i).folder, '/', train_info(2 + i).name));
    for j=1:length(tem_info) - 2
        [voice_data, Fs] = audioread(strcat(tem_info(2 + j).folder, '/', tem_info(2 + j).name));
        if j==1 
            mfcc_features = get_features(voice_data, Fs);
        else
            mfcc_features = [mfcc_features;get_features(voice_data, Fs)];
        end   
     GMM_features{i} = mfcc_features;  
    end
end
%
%Ä£ÐÍÑµÁ·
GMM_models = cell(1, speakers_num);
options = struct('MaxIter',{2000});
epochs = 10;
for i=1:speakers_num
    GMM_models{i} = fitgmdist(GMM_features{i}, GMM_num, 'RegularizationValue', 0.001, 'SharedCov', true, 'Options', options, 'Start', 'plus', 'Replicates', epochs);
end