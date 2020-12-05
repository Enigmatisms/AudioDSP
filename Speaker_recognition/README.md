​	本模型为GMM—MFCC模型，应用的方法为先下载到本地，在本地的Speaker Recognition中，建立两个文件：/train，/test，在/train文件夹中，不同的说话人再建立不同文件夹，其内添加.wav文件，在/test的文件夹中，添加.wav文件。

​	在运行前，请先加voicebox库添加进入路径。

​	在训练完，后建议保存工作区为mat文件，之后每次只需load模型参数进来就可以。

