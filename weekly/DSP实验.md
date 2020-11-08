<center><font size = 6 color='black'>DSP<font face='新宋体'><b>时域法分析报告</b></font></font></center>

---

<center><font size = 2>小组成员：何千越 *，张颖 *，米唯实，杨恺，安若鼎</center></font>

<center><font size = 2>（标有星号的成员参与了对应任务，未标星号的成员参与实验其他部分）</center></font>

**摘要** 实验一采用了基于AR模型的LPC-Kalman滤波进行的前置白噪声消除，此后输入系统中进行第一步的双门限粗分割，得到语音段大概位置，后对双门限结果采用最大类间方差分割（OTSU法）以及模拟退火算法进行端点优化。分割的结果使用两种不同的分类器：随机森林以及RBF核的SVM分别进行了训练与判定。时域法的准确率结果为79%(SVM)与80%(随机森林)，速度为（不含前置滤波时）6s同时处理8个音频，8个音频每个包含大约15个数字（采样率由48000Hz降到了22050Hz）。

**关键词：**Kalman滤波，LPC，自适应双门限，最大方差分割，模拟退火算法 

---

<font size = 5> **1   引 言 **</font>

​		在进行数字估计时，数据的输入可以有多种形式，比如数据集统一为一段音频只发声一次（比如人工智能班数据集），也可以使用实时输入的办法；当然由不同的需求，还可以对某一段音频进行离线处理，一段音频中包含多个数字。不管是以上的哪一种情况都涉及到一个问题：输入的语音段并非每一个数据点都存在有意义的信息。比如可能存在无人说话的部分，或是纯噪音的部分。我们将不同的语音段其定义为如下三种：

1. **有效段**：有着实际意义的语音段，比如包含了一个数字，注意有效段可以含噪声。
2. **噪声段**：没有实际意义，但是存在不近似为0的能量 / 过零率信息
3. **静音段**：几乎没有能量 / 过零率信息

​		由于输入数据存在噪声段以及静音段，导致我们无法直接将输入数据放入分类器中进行分类。输入数据不经过预处理将可能存在如下三个问题：

1. 数据虽然是有效段，但本身是含噪的，含噪语音作为训练集将会导致噪声形式无法完全涵盖的问题。
2. 数据是噪声段，虽然含有信息，但不包含任何一个数字
3. 数据是有效段，但是包括太多无效信息

​		以上两种情况都将会导致输入分类器的数据包含噪音或者无效信息。在进行机器学习时，分类器将会学习到一些不应该出现的特征，例如前后含有长静音段（欠分割），过零率异常（噪声过大）等等。进行验证集分类时的表现就会随之下降。基于以上的考虑，我们构建了一套时域预处理 / 分割 / 分类系统进行中文语音数字识别。

<font size = 5> **2  系 统 综 述**</font>

​		分割系统的流程图如下所示：

![](C:\Users\15300\Desktop\数电实验\DSP 实验1\flow.JPG)

<center><font size = 1>图1 系统流程图</font></center>

​		其中，LPC Kalman滤波内部的逻辑在系统总流程图的上半部分显示。在系统流程图中对应【噪声滤除】环节后的【LPC Kalman滤波】。系统凭借着：可选的前置滤波以及自适应双门限得到输入音频的分割。本小组使用的音频暂时是一段长音频，每段大概包含15个相同数字。系统分割的输出直接提供给分类器进行训练/预测。本系统使用Python / Cython [<font color = 'purple'>[Cython在本系统中的使用]</font>](#cython)混合实现。

​		第三部分主要是针对系统的原理 / 实现介绍：3.1 小节将介绍LPC Kalman滤波器对白噪声的剔除作用。3.2 小节将介绍基于双门限的VAD。3.3小节将介绍自适应优化。 3.4小节将介绍分类器。 3.5小节将介绍将介绍算法的加速优化。第四部分中，我们将展示时域法的结果。第五部分中得到我们的结论。

<font size = 5> **3  系 统 原 理 及 实 现**</font>

<font size = 4> **3.1  LPC Kalman滤波**</font>

​		Kalman滤波（以下简称KF）作为经典的线性高斯系统滤波器，在高斯噪声的滤除以及信息融合方面得到了广泛应用。最为广泛运用的KF为EKF（Extended Kalman Filter），其被广泛运用于视觉SLAM算法中，对于深度信息的滤波确定。本实验中，KF用于可能的语音段白噪声滤除。

​		我们讲说话人所处环境中存在的噪声视为白噪声。虽然在现实世界中，几乎不存在完美的白噪声，大多数噪声都是有色的，但考虑到白噪声能覆盖相当一部分情景（如下雨，信号不良）以及效应（低信噪比下传输引入的噪声），对白噪声进行分析以及处理是非常必要的。并且，如果需要对所有噪声都能有较为普适的处理方式，我们将需要实现【粒子滤波（Particle Filter）】（理论上PF可以对任意概率分布进行滤波），但粒子滤波的实现难度较大，并且速度显著慢于KF。出于“完全使用时域的方法进行实现”的考虑，在实验一中我们没有使用任何频域的信息（以及变换），在不适用频域分析的情况下很难进行FIR滤波器设计，于是我们使用了KF作为概率滤波器进行白噪声滤除。

​		由于人的语音信号中，波形前后是具有一定关联性的，但跨度过大的两个点的相关程度可能将非常小。对此我们采用了短时分析的手段，首先将输入数据（一般为一段采样率为48000Hz，包含约15个数字的音频）进行重采样，减小数据处理量。之后进行分帧操作，每一帧的长度固定，并且两帧之间的帧移为100%。由于在时域滤波过程中无需考虑平滑问题，不需要加强周期性，也无需进行帧间平滑，这使得加窗显得不那么必要。经过分帧处理之后，我们将对每一帧进行KF操作。需要注意的是，虽然我们的帧移为100%，但在进行滤波时，第i+1帧将会使用到第i帧的结果。这样可以保证我们的滤波结果是平滑的。

​		对于一帧语音，当其足够短时，我们可以将其使用**AR模型（auto-regressive）** **[[1]](#paper)** 进行建模。我们可以将人的声带看成一个AR模型下的系统，在很短的一帧时间内，当声带接收到一个刺激将会产生响应。由于短时不变性，可以近似认为，声带系统为一个线性时不变系统。那么发声的过程可以看作是：单位脉冲序列（人的气道气流）进入系统（喉头、声带），由于系统特性（声带周围肌肉对发声部位的控制）产生输出（人声）。则此短时近似的线性时不变系统可以写为如下形式**[1]**
$$
x(n)=\sum_{i=1}^Na_ix(n-i)+e(n)
$$
​		其中，$a_i$为自递归系统的系数，N为自递归系统的阶数，通俗来讲就是本时刻输出最多和N个时刻之前的输出存在关系。由上式可以得到误差$e(n)$的定义，出于对系统进行最优估计的考虑，我们希望实际的输出$x(n)$与近似系统预测$\hat x(n)$的均方误差（MSE）能够最小。而由于$x(n)$是一个随机过程，对MSE存在一个期望值估计：
$$
E(e)=\sum_n[x(n)-\hat x(n)]^2=\sum_n[x(n)-\sum_{i=1}^Na_ix(n-i)]^2
$$
​		对系统参数$a_i$求偏导，让$E(e)$取最小值并将平方式展开可以得到如下公式：
$$
\begin{align}
	&\sum_nx(n)x(n-j)=\sum_{i=1}^N\sum_nx(n-i)x(n-j)	\\
	&E=\sum_n[x(n)]^2-\sum_{i=1}^N a_i\sum_nx(n)x(n-i)
\end{align}
$$
​		此后化为Yule-Walker方程形式，并拆写为Toeplize矩阵，使用Durbin算法就可以求出滤波器的系数$a_i$。此即为对一帧时间内的响应进行了系统的LPC求解**[[2]](#paper)**。求解LPC的目的是：KF在迭代过程中需要状态转移以及对于噪声大小的估计。在本实验中，由于我们对系统所建的模型是一个高阶自递归的模型，状态转移的一阶Markov性也不再成立。从直观上来说，人声波形在某一时间点上的值不会只由其之前的一个值所决定（由于人声的复杂性）。高阶的Markov性表示我们需要对KF的形式进行一定的构造。由N阶AR模型递归表达式可以得到如下矩阵等式：
$$
\begin{equation}	
	\begin{pmatrix}
	0&1&0&0&...&0\\
	0&0&1&0&...&0\\
	0&0&0&1&...&0\\
	&...&&&...\\
	0&0&0&0&...&1\\
	a_{N}&a_{N-1}&a_{N-2}&a_{N-3}&...&a_{1}
	\end{pmatrix}
	*
	\begin{bmatrix}
	x_{n-N}\\x_{n-N+1}\\x_{n-N+2}\\...\\x_{n-2}\\x_{n-1}
	\end{bmatrix}
	=
	\begin{bmatrix}
	x_{n-N+1}\\x_{n-N+2}\\x_{n-N+3}\\...\\x_{n-1}\\x_{n}
	\end{bmatrix}\\
	
	\begin{pmatrix}
		\pmb 0_{N-1}&\pmb I_{(N-1)\times(N-1)}\\
		a_N&\pmb A_{N-1}^T
	\end{pmatrix}
	\pmb {x_{n-1}}=\pmb {x_n}
\end{equation}
$$
​		则公式（5）中的$N\times N$矩阵就是我们所需要的状态转移矩阵。而每一次通过LPC可以解出此状态转移矩阵，并且可以通过对$x_n$的估计$\hat x_n$计算出估计的误差，从而得到我们对于系统噪声的估计。由此，完整的KF流程可以写出：
$$
\begin{align}
	&\breve{x}(n)=A(n)\hat{x}(n-1)+e(n)\\
	&\breve{P}(n)=A\hat{P}(n-1)A^T+CR(n)C^T\\
	&K=\breve{P}(n)H^T(H\breve{P}(k)H^T+R)^{-1}\\
	&\hat x(n)=\breve{x}(n)+K(z(k)-H\breve{x}(k))\\
	&\hat P(k)=(I-KH)\breve{P}(k)
\end{align}
$$
​		其中所有的上标$\lor$符号表示了先验（prior），上标^表示了后验（posterior）。P代表状态协方差，K代表了Kalman增益，H代表了观测矩阵。C代表了一个除了最后一个元素为1以外，其余元素全为0的N维列向量（使用C是为了将LPC估计的误差送入状态估计协方差中，状态估计协方差是个矩阵，则尾部 $CR(n)C^T$也应该是个矩阵）。注意LPC与KF是交叉迭代的。对于每一帧，需要使用LPC / KF进行三次迭代。（可以看作是三个级联的LPC / KF滤波器）。实验表明，当只迭代两次时，静音区还是能够看到较为明显的噪声，而迭代三次及以上时，噪声几乎完全消失，并且语音段原有的噪声都可以进行一定的滤除（白噪声是人为加入的）。需要说明的是，除了数字1之外，白噪声滤除的效果非常好，播放消噪后语音段，人耳很难判定是否失真。LPC-KF结果如下图所示：

| <img src="C:\Users\15300\Desktop\数电实验\DSP 实验1\Figure_1.png" style="zoom:50%;" /> | <img src="C:\Users\15300\Desktop\数电实验\DSP 实验1\Figure_2.png" style="zoom:50%;" /> | <img src="C:\Users\15300\Desktop\数电实验\DSP 实验1\Figure_3.png" style="zoom:50%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                       原音频（数字0）                        |                          加入白噪声                          |                          滤波后音频                          |
| <img src="C:\Users\15300\Desktop\数电实验\DSP 实验1\Figure_4.png" style="zoom:50%;" /> | <img src="C:\Users\15300\Desktop\数电实验\DSP 实验1\Figure_5.png" style="zoom:50%;" /> | <img src="C:\Users\15300\Desktop\数电实验\DSP 实验1\Figure_6.png" style="zoom:50%;" /> |
|                       原音频（数字4）                        |                          加入白噪声                          |                          滤波后音频                          |

<font size = 4> **3.2 双\门限的VAD**</font>

(1)  特征选取

​		由于表征语音信号特征的各参数是随时间而变化的，所以它是一个非平稳态过程，无法用数字处理技术进行处理，因此对其特征的提取应采取“短时分析技术”，即将整段语音信号先进行短时分帧，在每一帧内可将其近似看作准稳态过程，计算其对应短时特征参数值。

​		① 短时能量（区分浊音和静音）

​		设第n帧语音信号$x_n(m)$包含N个采样点，其短时能量用$E_n$表示，则其计算公式如下：
$$
E_n=\sum_{m=0}^{N-1}x_n^2(m)
$$


​		② 短时过零率（区分清音和静音）

​		短时过零率表示一帧语音中语音信号穿过横轴（零电平）的次数，对采样后的离散信号，过零率即为样本改变符号的次数。

​		设第n帧语音信号$x_n(m)$包含N个采样点，则定义其短时过零率$Z_n$定义为：
$$
Z_n=\frac 1 2\sum_{m=0}^{N-1}|sgn[x_n(m)]-sgn[x_n(m-1)]|,\; where\\
sgn[x]=
\left\{
	\begin{array}{**lr**}
		1, (x\ge 0)\\
		-1,(x\lt 0)
	\end{array}
	
\right.
$$
​		

​	其中sgn为符号函数，定义如上

(2) 双门限法基本步骤

![](C:\Users\15300\Desktop\数电实验\DSP 实验1\vad.png)

​		① 按帧计算信号的短时能量和短时过零率，由所得特征参数时间序列画出整段语音信号特征参数的变化图像。

​		② 根据语音能量的轮廓选取一个较高的门限$T_2$，使得语音信号的能量包络大部分都在此门限之上，根据该门限值进行一次初判。如图所示，判得高于该阈值的CD段一定为语音段，而语音起止点位于该门限与短时能量包络交点C和D所对应的时间间隔之外。

​		③ 观察语音信号波形特征估计出前导无话段的长度，假设整段语音信号噪声是平稳的，则可以利用前导无话段的参数特征估算噪声的特性，根据背景噪声的能量确定一个较低的门限$T_1$，并从初判点起点往左，从初判点终点往右搜索，分别找到短时能量曲线第一次与该门限的交点B和E,则BE段即为判断所得的浊音段。

​		④ 根据噪声的过零率特征确定门限值$T_3$，再从B点向左，E点向右进行搜索，找到短时平均过零率低于阈值$T_3$的两点A、F。另外考虑到语音发音时单词间存在停顿时间，因此设定一个最大静音长度Maxsilence，只有当F点之后语音段过零率小于$T_3$的时间大于Max silence时才判断语音段结束，这样可以避免对同一组发声产生多个分割段输出，实际相当于延长了尾音的长度。则最终所得语音有效段为AF。

​		⑤ 在判断出一段满足上述筛选条件的语音段后，还需要将其持续长度与语音有效段最小长度Minlen作比较，若长度过小，则将该段视为噪音舍弃。

(3) 算法实现

​		① 参数说明(Matlab实现版本)

​		基于短时能量和短时过零率的双门限算法通过调用函数vad1实现，调用格式为[voiceseg, vs1, SF, NF] = vad1(x, wlen, inc, NIS)，其中输入量x表示采样点信息存储矩阵，wlen表示帧长，inc表示帧移，NIS表示所定义的前导无话段的长度，返回值voiceseg是一个存储语音有效段起点，终点，持续长度的结构数据，vs1为有效段数量，SF, NF用于区分有话帧和无话帧。

② 流程图

<img src="C:\Users\15300\Desktop\数电实验\DSP 实验1\flow2.png" style="zoom:60%;" />

(4) 缺陷及基于普通双门限法的改进

**缺陷：**

​		① 在计算过零率时，由于语音环境复杂，信号可能会有微小的零漂，使得在短时过零率特征上无话段与有话段区别不明显。

​		② 噪声情况下短时过零率变化较大，表现为在无话段的过零率比有话段还要大，此时按传统双门限法判别可能将整个噪声区域都作为语音信号被选中，而语音部分却被反选为无话段，因此此时传统的双门限方法不再适用。

​		③ 算法假设整个语音段的噪声是平稳的，即前导无话段噪声特性对整段语音都适用，实际情况显然比这复杂。

​		④ 函数中amp1, amp2, zcr2的设置虽然是根据前导无话段噪音动态变化的，但是其比例数却是固定的，因此在不同噪声和信噪比环境下自适应性不强。另外Maxsilence,Minlen设为了定值，但实际上为了保证检测准确性它们需要根据语速进行调整。

**改进：**

​		① 为了消除零漂，可以先对语音信号进行中心截幅处理，即消除一定的直流分量后再计算过零率。

​		② 在噪声情况下对过零率门限条件作适当修改，即将短时过零率小于门限的帧记作有话帧。

<font size = 4> **3.3 双门限VAD自适应优化**</font>

​		双门限法直接分割得到的结果并不够好，在之前的周报中我们也曾提到，双门限法的核心在于：对噪声能够有一个较好的估计。但显然，噪声端的位置是很难估计的，切割语音段以及确定噪声段就是两个互补的问题。如果对噪声能有较好的估计，那么门限设置将能得到较好的分割。但双门限法做了一个非常不好的假设：起始段存在一段静音（底噪内容）。而在3.2节中，我们已经提到，基本双门限VAD使用的不是固定的阈值，我们使用了一个根据噪声自适应的阈值，虽然一定程度上能对噪声环境自适应。<u>**但其实这样也是存在问题的**</u>，由于数据获取的方式不一，很可能存在：

1. 起始段已经存在了人声
2. 起始段完全静音
3. 起始段噪音与其他段噪音不匹配

​		上述情况均会导致门限的估计错误，以致于最终得不到正确的结果，甚至出现程序bug崩溃的情况。更重要的是，**使用这种对所有数字均固定的阈值没有对有效数据自适应的能力**。这种单一的双门限阈值，导致我们需要对每一个数字进行折衷，不管对应语音段长 / 短，音量大 / 小，声音尖锐 / 沉闷。这种折衷必然导致分割精度的下降。在此我们使用了两种方法进行优化：

(1) OTSU阈值（大津法）

​		这个方法实际上来自于CV领域。问题背景是：我们需要一个阈值，能够分离前景和背景，前景和背景组成的图像灰度分布是单峰的。对前景以及背景最优的分割代表着：前景和背景两个子集之间的**类间方差最大**。本问题中类似，假设有效段为前景，噪声段和静音段为背景（底噪）。我们先使用双门限法获得粗分割（已经说明，粗分割需要对不同性态的语音端进行降低精度的折衷），对于粗分割后的一段粗分割语音，假设一共存在$M$帧分帧后的数据，其中包含$\omega_1$比例的前景帧（平均幅值大于等于阈值），$\omega_2$的底噪帧（平均幅值小于阈值）。我们可以求出前景帧的平均幅度$\mu_1$，以及底噪帧的平均幅度$\mu_2$。那么可知，整段粗音频的平均幅值$\mu$应该为:
$$
\mu=\omega_1\mu_1+\omega_2\mu_2
$$
​		类间的方差可以求出，如下式所示。物理意义为：前景帧与平均幅度的偏差平方和 + 底噪帧的平均幅度的偏差平方和。
$$
\sigma^2=\omega_1(\mu-\mu_1)^2+\omega_2(\mu-\mu_2)^2
$$
​		当$\sigma^2$最大时，其意义为：前景与背景的差异最大，得到了最小二乘意义上的最优分割。而很可惜的是，由于阈值是个非线性滤波器，不存在求导优化的方式，只能通过遍历（或者给定初始估计以及范围的遍历）进行求解。那么显然，此阈值分割必须在数字域上进行。我们必须将浮点表示的音频平均幅值进行归一化，并量化到0~255这256个值上（与图像处理一致）。注意由于我们在进行OTSU分割时，是对每一粗分割段进行处理，显然在这种情形下，计算方差与均值的计算量会比对完整音频小，速度快。并且由于段与段之间互不影响，我们的处理可以进行并行化。

(2) 退火算法端点优化

​		粗分割会导致两个问题：(1)端点延伸过大，包含过多静音段或者底噪段 。(2)如果说话人在数字说完之后出现向麦克风呼气的情形（吹麦声），双门限法一般会认为这属于有效音频（尽管其幅值相对小，并且存在过渡段）（见下图）。显然对于第一段音频，普通双门限法的估计是完全不行的，包含了过多无效信息，在分类器学习的过程中将引入不应该出现的特征。

<img src="C:\Users\15300\Desktop\数电实验\DSP 实验1\norm_vad.png" style="zoom:80%;" />

​		OTSU阈值分割在噪声较小时，一般可以有效地将噪声划分在底噪区。但是它也存在如下问题：(1) 对于3 / 4 / 7这样存在清音的数字，很可能将摩擦音划分为底噪。(2) 如果双门限结果较好，含有的底噪很少，那么阈值分割将会将部分有效段切分到底噪段（可想而知，有段内进行OTSU必然会造成过度分割）。在进行OTSU优化之后，很可能端点会在有效段内。那么我们需要使用退火算法进行端点优化。

​		使用退火算法的原因是，幅度曲线在划分的端点处很可能不是平稳下降的。如果我们只简单地考虑，让后端点向后搜索，更低位置就接受，那么若遇到小的峰值（甚至突起）将无法越过。若采用退火算法，按照Metropolis准则接受移动，则在概率上能够以更大的期望移动到正确的端点值。退火算法还有一个好处就是，由于其接受概率随着移动次数减小，在接受上升移动时也与上升量成反比，若噪音较大，退火算法将不会翻越到噪声段的后部。对于后端点优化，算法在执行时先尝试向前搜索可能的下降。当最终下降达到一定程度时（比如比初始幅度值的0.5倍还小），我们将接受这个解，不进行向后的退火搜索。当向前搜索失败，则进行向后的搜索。仍是以上图所对应的数字为例：这是一串噪声比较严重的数字1（**说话人在语音结束后对麦克风呼气，由频谱分析可以看出后段的过零率很大，频谱分布均匀**）。下图为未使用退火算法端点优化的结果（直接OTSU）与退火算法优化的结果：

| <img src="C:\Users\15300\Desktop\数电实验\DSP 实验1\direct.png" style="zoom:50%;" /> | <img src="C:\Users\15300\Desktop\数电实验\DSP 实验1\right.png" style="zoom: 50%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                         OTSU直接分割                         |                         退火算法优化                         |

<font size = 4> **3.4 分类器**</font>

​		处于不想让系统过于复杂的考虑，我们主要使用了两个分类器进行对比：

1. 随机森林

    		2. RBF核的SVM

​		关于这两个分类器的原理，就不在此赘述了，数字信号处理实验，重点应该是**数字信号【处理】**。但需要提及的是两种机器学习算法各自的优点以及我们选这两种算法分别的初衷。随机森林的优点在于其更不容易产生过拟合，并且对数据噪声不敏感。对于我们的切割数据，由于双门限切分中引入了退火算法（随机优化），即使是相同的数据，两次切割结果可能也不相同。我们期望分类器能对同一段数据的小幅度变化鲁棒。而RBF核函数的SVM中小样本的分类准确度很高，并且有着深厚的理论基础。

​		对于两个分类器，我们最终输入的特征都是相同的。构造特征向量的过程是：首先进行长度上的归一化，也即无论切割的这一段音频长度为多少帧，我们都将其分帧为固定且相同的帧数（如本实验中我们分帧全部分为50帧）。对于每一帧，产出3个特征。这样50帧可以一共得到150个数据点。而最后我们使用LPC的方式对整段音频每10帧求取一次LPC系数，产出共50个LPC系数，一共200个特征。

| 在特征向量中的位置 | 特征的意义                                                |
| ------------------ | --------------------------------------------------------- |
| 0~49               | 按最大值归一化平均幅度                                    |
| 50~99              | 按最大值归一化的平均过零率                                |
| 100~149            | 按最大值归一化的平均幅值差（取代自相关）**[[3]](#paper)** |
| 150~199            | LPC系数                                                   |

​		需要说明的是平均幅值差与LPC系数的计算。由于50帧中，每一帧只能产出一个平均幅值差信息，我们无法求取每一种时间差对应的平均幅值差，最终的选择是求取半帧长度的平均幅值差。而LPC系数的计算已经提到，50帧中，每5帧为一组，进行5阶LPC系数的求取（相当于求10帧短时对应的声门系统的5阶自递归系数），可以得到50个特征点。组织成4*50的矩阵，进行ravel()操作放入分类器中训练。

​		最后我们的数据集大概为：训练集每个数字有约90个，验证集每个数字约15个。

<font size = 4> **3.5 算法加速**</font>

​		经过实测，我们发现其中存在一些非常慢的算法实现，包括LPC-Kalman滤波。KF在本实验中非常慢几乎是必然的，对于每一个点都需要进行三次KF计算（三次迭代），每一次计算至少涉及到2次(16$\times$16)矩阵的乘法与加法，尽管我们计算出的状态转移矩阵存在稀疏性以及一定的规律性，我们曾经尝试过利用稀疏性，使用Python数组切片进行矩阵的分块运算，希望减少计算量，但结果比直接矩阵运算更差了，个人认为可能numpy库内部优化做的很好，大型矩阵运算会比数组切片快很多。

​		第二个较慢的操作是librosa库的wav音频读取，由于我们在读取过程中，对wav文件进行了一个48000Hz到22050Hz的降采样。降采样操作非常耗时，平均可以占音频长度的1/10时间，比如一个18s的音频，平均使用1.8s才能进行读入。但librosa库读入的数据比较精确，虽然我们使用wave库取代librosa进行读取，可以让读取快1000倍，但是我们发现，在其量化时会进行一个类似于调制的操作（个人认为应该就是调制，出现了大范围的波动，低频大范围的波动内包含了很多内部高频振动），这会导致过零率出现异常，导致双门限法分割出现错误。

<span id='cython'>

​		此外，双门门限法也存在时间较长的运算。比如过零率计算，阈值分割与退火算法（优化）。我们使用了两个手段进行加速：

- Cython加速。Cython是一种预编译技术，将Python标准文件.py使用.pyx进行编写，在其中可以使用Cython的语法。我们知道，Python便于新手学习的一个原因是其动态性，所有的数据类型都是可变的，在运行解释阶段需要一级一级地查其类型。但这也是其运算慢的一个原因，其数据类型不是静态的。Cython中使用如下语法（例子）:

```cython
cdef numpy.ndarray[float, ndim=1] array = np.zeros(100)
```

​		可以定义一个确定了数据类型以及维度的numpy数组。这将加快其处理。并且使用Cython可以对部分Python程序进行预编译，得到一个库(.pyd)，直接import此库相当于调用了一个Python内置库，在解释阶段不需要对内置库的函数进行解释，可以节省解释的时间。Cython的编译器是基于mingw64的gcc编译器。

- 多进程并行计算。多线程并不能完全使用CPU资源，在查看CPU占用率之后发现，CPU在多线程模式下只使用了CPU单核进行处理。但多进程（Multiprocess）可以完全使用CPU资源（100%CPU利用率）。现阶段，我们使用Cython + 多进程技术，在6s内可以处理完8段音频，音频总共包含平均90个数字，48000Hz采样率，总长度约120s（每个数字大约1.2~1.5s）。对于加速的结果，**见3.4 实验结果**。

</span>

<font size = 5> **4 实 验 结 果**</font>

随机森林分类结果：

```Python
Random Forest Classifier:
Predicted result:  [0 0 5 0 0 0 8 8 2 8 0 0 6 0 0 0 0 0 0 0 0 0 0 0 0 0 6 5 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 5 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 8 2 2 2 8 2 2
 2 2 2 2 2 2 2 2 3 3 3 3 3 3 9 4 3 9 3 3 9 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 7
 4 7 4 4 0 4 3 2 7 3 9 4 4 4 4 4 4 4 4 4 4 4 3 4 3 5 5 5 1 5 5 5 5 1 5 5 5
 5 5 5 5 5 5 6 5 5 5 5 5 5 5 5 5 1 6 6 1 6 6 2 8 2 6 0 0 1 6 6 6 6 6 6 6 6
 6 0 6 6 6 6 6 7 9 1 7 7 7 9 7 7 7 7 7 9 7 7 7 4 7 7 7 7 7 7 7 7 4 7 7 8 8
 8 8 8 8 8 8 8 8 2 8 8 8 2 8 8 8 8 8 8 8 8 2 2 2 8 9 9 9 9 9 9 9 9 9 9 9 9
 9 9 9 3 4 3 9 4 9 3 4 3 9 9 9]
While truth is:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4
 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5 5 5 5
 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 8 8
 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 9 9 9 9 9 9 9 9 9 9 9 9
 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9]
False Ratio:  0.20072992700729927
```

SVM（RBF核）的分类结果

```python
SVM with RBF kernel:
Predicted result:  [0 0 6 0 6 3 6 0 2 0 0 0 6 0 0 0 0 0 0 0 0 0 0 0 1 6 2 2 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0
 2 2 2 2 2 2 2 2 3 3 3 9 3 3 0 4 4 9 3 4 3 3 3 3 3 3 3 9 3 3 3 3 3 3 3 4 3
 4 7 4 4 0 4 4 0 4 0 4 4 4 4 4 4 4 4 4 4 4 4 3 4 3 5 5 5 5 5 5 5 0 1 5 8 5
 5 5 8 5 5 5 8 5 5 5 8 5 5 5 2 5 0 0 6 6 6 6 6 6 0 0 0 0 1 0 0 6 6 6 6 0 6
 6 0 6 6 6 6 6 7 7 7 7 7 7 9 7 0 7 7 7 9 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 8 8
 5 8 8 5 8 8 1 0 0 8 8 5 8 8 8 2 8 8 8 8 8 8 8 2 8 9 9 9 9 9 9 9 9 9 9 9 9
 9 7 9 9 4 4 9 4 9 3 9 9 9 9 9]
While truth is:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4
 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 5 5 5 5 5 5
 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6
 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 8 8
 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 9 9 9 9 9 9 9 9 9 9 9 9
 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9]
False Ratio:  0.21897810218978103
```

速度以及CPU使用分析（可以明显看出CPU使用率由于多进程达到了100%）：

<img src="C:\Users\15300\Desktop\数电实验\DSP 实验1\speed.JPG" style="zoom: 36%;" />

<font size = 5> **5  结 论**</font>

​		在实验一（时域法语音数字识别）中，我们通过LPC-KF技术可以对输入数据进行现行的滤波，对于白噪声的消除能力较强，但LPC-KF存在速度低，难以满足实时性处理的问题（LPC-KF没有使用任何速度优化技术）。在进行可选的前置滤波之后，我们使用了对噪声自适应的双门限算法进行粗分割，粗分割由于存在对于所有语音段进行门限的折衷问题，分割效果不够好。我们先是使用OTSU阈值对平均幅度进行自适应阈值分割，由于此算法会导致幅度小的清音被忽略，以及双门限法分割效果较好时将会收束分割位置，我们需要退火算法对分割位置进行搜索优化。最后得到较为准确的分割结果，送入分类器。分类器对于分割数据的准确性为80%左右，其中可能存在一定的过拟合问题，还需要扩大数据集进行学习 / 训练。我们已经从人工智能班取得他们的共享数据集，但时域法还没有开始测试。对于实验的代码实现，我们使用了两种方法进行加速，使原来串行处理时18s才能处理完一个音频优化到6s可以同时处理完8个音频。当然，本实验中还存在很多可以优化的方面，由于时间关系无法一一完成。

<font size = 5> **6  参 考 文 献**</font>

<span id ='paper'>

[1]M. Markovic, "On determining heuristically decision threshold in robust AR speech model identification procedure based on quadratic classifier," *ISSPA '99. Proceedings of the Fifth International Symposium on Signal Processing and its Applications (IEEE Cat. No.99EX359)*, Brisbane, Queensland, Australia, 1999, pp. 131-134 vol.1, doi: 10.1109/ISSPA.1999.818130.

[2] 熊琦 ，杜旭 ，朱晓亮，一种基于短时平均幅度差的语音检测算法， 语音技术，文章编号：1002-8684（2006）09-0050-04

[3] Y. Chan and R. Langford, "Spectral estimation via the high-order Yule-Walker equations," in *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 30, no. 5, pp. 689-698, October 1982, doi: 10.1109/TASSP.1982.1163946.

</span>

<font size = 5> **7  附 录**</font>

​		也可以直接转到[【链接🔗：Github:AudioDSP Repository】](https://github.com/Enigmatisms/AudioDSP)查看相关代码。

​		Python代码：

1. 特征提取与训练

```Python
"""
    尝试提取时域特征
    尝试python 并行计算
        由于每一帧特征提取得到的结果都是相互独立的
    个人觉得特征可以使用如下方式组织：
        1. 每帧的短时幅值 比如我们固定地对某一语音信号分成56帧，那么前两个特征共112阶，只需要一个16阶LPC就可以得到最后的特征
        2. 每帧的短时过零率
        3. 每帧的LPC预测系数
        每一个特征向量长128，最好不要超过256，否则个人认为分类器负担太重
    首先是分割操作，个人先随便设一个阈值吧
    对分割好的纯语音信号进行分帧加窗，帧移大概为10%
    此后进行特征提取，放入分类器进行训练
    测试集进行验证
"""

import os
import librosa as lr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
import pickle as pkl

# 无加窗操作的framing
def windowedFrames(data, fnum = 50):
    length = data.size
    flen = int(np.floor(length / 0.9 / fnum))
    frames = np.zeros((fnum, flen))
    shift = int(0.9 * flen)
    # print(frames.shape, flen, shift, data.size)
    for i in range(fnum):
        if i * shift + flen > length:
            frames[i, : length - i * shift] = data[i * shift:]
        else:
            frames[i, :] = data[i * shift: i * shift + flen]
    return frames
    
# 得到一个fnum * 1维向量，为每一帧的平均幅度
def averageAmplitude(frames, fnum = 50):
    res = np.zeros(fnum)
    for i in range(fnum):
        res[i] = np.mean(abs(frames[i, :]))
    res /= np.max(res)      # 最值归一化
    return res

# 平均过零率 向量(无需归一化）
def averageZeroCrossingRate(frames, fnum = 50):
    res = np.zeros(fnum)
    _, cols = frames.shape
    for i in range(fnum):
        cnt = 0
        for j in range(cols - 1):
            if (frames[i][j] > 0 and frames[i][j + 1] < 0 ) or (frames[i][j] < 0 and frames[i][j + 1] > 0):
                cnt += 1
        res[i] = cnt / frames.size
    return res

# 平均幅度差 (半帧 / 0.25 帧长 自相关)
def averageAmpDiff(frames, fnum = 50):
    _, cols = frames.shape
    half = int(cols / 2)
    dhalf = int(half / 2)
    res1 = np.zeros(fnum)
    res2 = np.zeros(fnum)
    for i in range(0, fnum):
        res1[i] = sum(abs(frames[i, :-half] - frames[i, half:]))        # 半帧帧移类自相关特征
        res2[i] = sum(abs(frames[i, :-dhalf] - frames[i, dhalf:]))      # 1/4 帧移类自相关特征
    res1 /= np.max(res1)
    res2 /= np.max(res2)
    return np.concatenate((res1, res2)) 

# 个人觉得这个特征很可能不行，而且还会很慢
def getLPC(frames, fnum = 50):
    step = int(fnum / 10)
    res = np.zeros(fnum)
    for i in range(0, 10):
        _lpc = lr.core.lpc(frames[i * step], 5)[1:]
        res[i * 5 : (i + 1) * 5] = _lpc
    res /= np.max(res)
    return res

# 如果是一段语音分为50帧，那么得到的结果就是一个长度为250的向量，大概为16 * 16图片的规模（不过是浮点数）
def getFeatures(frames, fnum = 50):
    amp = averageAmplitude(frames, fnum)
    azc = averageZeroCrossingRate(frames, fnum)
    aad = averageAmpDiff(frames, fnum)
    lpc = getLPC(frames, fnum)
    return np.concatenate((amp, azc, aad, lpc))

def loadWavs(head = "..\\full\\", fnum = 50):
    feats = []
    classes = []
    for num in range(10):
        directory = head + "%d"%(num)
        if os.path.exists(directory) == True:
            for i in range(105):                    # 最多105个
                path = directory + "\\%d.wav"%(i + 1)
                if os.path.exists(path) == False:
                    break
                data, sr = lr.load(path, sr = None)
                data = lr.resample(data, sr, 8000)
                frms = windowedFrames(data, fnum)
                feats.append(getFeatures(frms, fnum))
                classes.append(num)         # 类别增加
                print("Loading from " + path)

    feats = np.array(feats)
    classes = np.array(classes)

    return feats, classes

if __name__ == "__main__":
    use_forest = False
    fnum = 50
    C = 100
    gamma = 0.001
    max_iter = 2000

    load = False             # 是否加载训练集（是否使用保存的模型）

    test_data, test_label = loadWavs(head = "..\\full\\c")
    if load == True:
        train_data, train_label = loadWavs(fnum = fnum)
        if use_forest:
            clf = RFC(max_depth = 16, min_samples_split = 6, oob_score = True)
            clf.fit(train_data, train_label)
        else:
            clf = SVC(C = C, gamma = gamma, max_iter = max_iter, kernel = 'rbf')
            clf.fit(train_data, train_label)
        if use_forest:
            path = "..\\model\\forest.bin"
        else:
            path = "..\\model\\svm.bin"
        with open(path, "wb") as file:      # pickle 保存模型
            pkl.dump(clf, file)
    else:    # pickle 加载模型
        if use_forest:
            path = "..\\model\\forest.bin"
        else:
            path = "..\\model\\svm.bin"
        with open(path, "rb") as file:
            clf = pkl.load(file)
    res = clf.predict(test_data)
    print("Predicted result: ", res)
    print("While truth is: ", test_label)
    print("Difference is: ", res - test_label)

    rights = (res - test_label).astype(bool)
    print("Right Ratio: ", rights.sum() / res.size)


    # plt.plot(np.arange(data.size), data, c = "k")
    # plt.show()
```

2. VAD门限与算法优化：

```python
"""
    类封装 / 多线程加速
    @author HQY
    @date 2020.10.28
"""

import os
import sys
import numpy as np
import multiprocessing as mtp
import matplotlib.pyplot as plt
from time import time
from librosa import load as lrLoad
from cv2 import THRESH_BINARY, THRESH_OTSU
from cv2 import threshold as cvThreshold
import wave
import audioop
from modules import *

class VAD:
    _fig_num = 1
    __maxsilence__ = 10
    __minlen__  = 12
    def __init__(self, file, wlen = 800, inc = 400, IS = 0.5):
        self.y = None       # 分帧结果
        self.zcrs = None
        self.amps = None
        self.ends = None
        self.starts = None
        if not os.path.exists(file):
            raise ValueError("No such file as '%s'!"%(file))
        start_t = time()
        self.data, self.sr = lrLoad(file)
        end_t = time()
        print("Data loaded. Time consumed: ", end_t - start_t)
        # 需要在此处load
        # 每个类对应处理一段语音

        self.N = self.data.size
        self.wlen = wlen
        self.inc = inc
        self.IS = IS
        self.NIS = int(np.round((IS * self.sr - wlen) / inc + 1))
        self.fn = int(np.round((self.N - wlen) / inc))

    # VAD 得到结果后需要进行进一步处理 退火式的搜索
    def _vadPostProcess_(self, prev = 8):
        threshold = self.amps[self.starts] / 4            # 1/4 当前门限
        for i, start in enumerate(self.starts):
            proba = 0.0
            step = 0
            for step in range(prev):
                if step == 2:       # 当step == 2 时开始引入随机拒绝性
                    proba = 0.001
                _amp = self.amps[start - step]
                if _amp < 0.01:
                    break
                if _amp > threshold[i]:   # 当大于门限时，step >= 2 有概率停止，step 越大越容易停止
                    if step >= 2 and np.random.uniform() < proba:
                        break
                    proba *= 1.5
                else:       # 小于门限直接停止
                    break
            self.starts[i] -= step       # 移动端点门限
            if self.starts[i] < 0:
                self.starts[i] = 0

    # 平均幅度的可视化
    @staticmethod
    def _averageAmpPlot_(amps, starts, ends, plot_diff = True, save = False, path = ""):
        plt.figure(VAD._fig_num)
        VAD._fig_num += 1
        fn = amps.shape[0]
        plt.plot(np.arange(fn), amps, c = 'k')
        plt.scatter(np.arange(fn), amps, c = 'k', s = 6)

        for start, end in zip(starts, ends):
            ys = np.linspace(-0.5, 0.5, 3);
            plt.plot(np.ones_like(ys) * start, ys, c = 'red')
            plt.plot(np.ones_like(ys) * end, ys, c = 'blue')
            amp_mean = np.mean(amps[start:end])
            plt.annotate("%.3f"%(amp_mean), xy = (start, 0.5), xytext = (start, 0.5))
        plt.title("Average Amplitude")
        if plot_diff:
            plt.figure(3)
            diff = amps[:-1] - amps[1:]
            diff.resize(diff.size + 1)
            plt.plot(np.arange(diff.size), diff, c = 'k')
            plt.scatter(np.arange(diff.size), diff, c = 'k', s = 6)
            for start, end in zip(starts, ends):
                ys = np.linspace(-0.5, 0.5, 3);
                plt.plot(np.ones_like(ys) * start, ys, c = 'red')
                plt.plot(np.ones_like(ys) * end, ys, c = 'blue')
                plt.title("Average Amplitude Difference")
        if save:
            plt.savefig(path)
            plt.cla()
            plt.clf()
            plt.close()

    # @jit(nopython = True)
    @staticmethod
    def _calculateMeanAmp_(y):
        wlen, fn = y.shape
        amps = np.zeros(fn)
        for i in range(fn):
            amps[i] = np.sum(np.abs(y[:, i])) / wlen
        return amps

    def _adaptiveThreshold_(self):
        for i in range(self.starts.__len__()):
            start = self.starts[i]
            if start < 0:
                start = 0
            end = self.ends[i]
            _amp = self.amps[start:end]
            _amp = _amp / max(_amp) * 255
            _amp = _amp.astype(np.uint8)
            _amp = cvThreshold(_amp, 0, 1, THRESH_BINARY | THRESH_OTSU)[1].astype(int)
            _amp = (_amp[:-1] - _amp[1:])
            _amp.resize(_amp.size + 1)
            end_counter = 0
            for j in range(_amp.size):
                if _amp[j] > 0:
                    if end_counter < 2:                         # 多峰时，最多移动两次终止点
                        end_counter += 1
                        self.ends[i] = start + j     # 精分割，并扩大一帧选区

    """
    退火搜索，想法是：向左右进行端点搜索，退火温度不要太高
        开始时对start - 20, end + 20 内的amps归一化
        最大迭代次数为20次：
        前6次最大步长为3帧，此后变为两帧（左右）
        较高的值按照概率接受（接受变化，但不接受作为最终值）
        较低的值分为正常、临界、异常值
            正常值（合适幅值）直接移动
            临界（处于过大与过小值之间的，难以判定值）：记录临界值数目，按照数目确定的概率接收 0.06~0.03
                临界值多了，接受概率就会变低
            异常值（过小值 < 0.03）：直接终止算法
        帧移不超过20帧（20帧对应了6000）（调整的最大限度）
        开始时就小于0.03的端点不优化
    """
    def _annealingSearch_(self, max_mv = 20):
        length = self.starts.__len__()
        for i in range(length):
            start = self.starts[i] - max_mv
            old_end = self.ends[i]                      # 记录原有的终止点
            end = self.ends[i] + max_mv
            if i == 0:      # 防止array越界
                start = max(0, start)
            elif i == length - 1:
                end = min(self.amps.size - 1, end)
            max_val = max(self.amps[start:end])
            _temp = self.amps[start:end].copy()
            _temp /= max_val                    # 归一化
            pos, reverse = invAnnealing(_temp, old_end - start, start_temp = 1.2)
            if reverse == True:
                self.ends[i] = start + pos
            else:
                print("Reverse is False.")
                self.ends[i] = start + annealing(_temp, old_end - start)  # 从原有的终止点开始搜索 返回值为移动的步长

    @staticmethod
    def _normalizedAmp_(amps, starts, ends):
        for start, end in zip(starts, ends):
            max_val = max(amps[start - min(15, start):end + 15])
            amps[start - min(15, start):end + 15] /= max_val
        plt.figure(VAD._fig_num)
        VAD._fig_num += 1
        plt.plot(np.arange(amps.size), amps, c = 'k')
        plt.scatter(np.arange(amps.size), amps, c = 'k', s = 6)
        for i in range(len(starts)):
            ys = np.linspace(0, 1, 3);
            plt.plot(np.ones_like(ys) * starts[i], ys, c = 'red')
            plt.plot(np.ones_like(ys) * ends[i], ys, c = 'blue')

    def reset(self, num = 1):
        VAD._fig_num = num

    def process(self, do_plot = True):
        self.y = enframe(self.data, self.wlen, self.inc)          # 分帧操作
        self.zcrs = zeroCrossingRate(self.y, self.fn)
        self.amps = calculateMeanAmp(self.y)              # 平均幅度
        # ================= 语音分割以及自适应化 ========================
        # self._vadSegment_()   
        self.starts, self.ends = vadSegment(self.y, self.zcrs, self.fn, self.NIS)
        self.starts, self.ends = faultsFiltering(self.amps, self.starts, self.ends, 0.012)
        self._vadPostProcess_(12)
        self._adaptiveThreshold_()
        self._annealingSearch_()
        # =============================================================
        if do_plot:
            print("Voice starts:", self.starts)
            print("Voice ends:", self.ends)
            print("VAD completed.")
            plt.figure(VAD._fig_num)
            VAD._fig_num += 1
            plt.plot(np.arange(self.N), self.data, c = 'k')
            for i in range(len(self.starts)):
                ys = np.linspace(-1, 1, 5);
                plt.plot(np.ones_like(ys) * self.starts[i] * self.inc, ys, c = 'red')
                plt.plot(np.ones_like(ys) * self.ends[i] * self.inc, ys, c = 'blue')
            VAD._averageAmpPlot_(self.amps, self.starts, self.ends, True)
            # print(self.amps.shape)
            VAD._normalizedAmp_(self.amps, self.starts, self.ends)
            plt.show()

def vadLoadAndProcess(path, do_plot = False):
    vad = VAD(path)
    vad.process(do_plot)
    

if __name__ == "__main__":
    number = sys.argv[1]
    using_thread = 0
    if sys.argv.__len__() == 3:
        using_thread = int(sys.argv[2])
    start_t = time()
    if using_thread:
        proc_pool = []
        for i in range(7):
            file = "..\\segment\\%s\\%s%02d.wav"%(number, number, i)
            pr = mtp.Process(target = vadLoadAndProcess, args = (file, False))
            proc_pool.append(pr)
            pr.start()
        file = "..\\segment\\%s\\%s%02d.wav"%(number, number, 7)
        vadLoadAndProcess(file)
        for i in range(7):
            proc_pool[i].join()
    else:
        file = "..\\segment\\%s\\%s02.wav"%(number, number)
        vadLoadAndProcess(file, True)
    end_t = time()
    print("Running time: ", end_t - start_t)
```

3. Cython 被加速算法：

```cython
cimport numpy as np
cimport cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def annealing(np.ndarray[np.float64_t, ndim=1] seg, int end, int max_iter = 30, float ab_thresh = 0.03, float start_temp = 0.3, int direct = 1):
    cdef int now_pos = end
    cdef int seg_len = len(seg)
    cdef int min_pos = now_pos
    cdef int tipping_cnt = 0
    cdef int i = 0
    cdef int step = 0
    cdef int tmp_pos = now_pos
    cdef float this_val = 0.0
    cdef float min_val = seg[now_pos]
    cdef float start_val = min_val
    for i in range(max_iter):
        if i > 6:
            step = np.random.choice((1, 2)) * np.random.choice((-1, 1))
        elif i > 2:
            step = np.random.choice((1, 2, 3)) * np.random.choice((-1, 1))
        else:
            step = np.random.choice((1, 1, 2, 2, 3))
        tmp_pos = now_pos + step * direct
        if tmp_pos >= seg_len:
            tmp_pos = seg_len - 1
        elif tmp_pos <= 0:
            tmp_pos = 0
        now_val = seg[tmp_pos]
        if now_val < min_val:       # 接受当前移动以及最终移动
            min_val = now_val
            now_pos = tmp_pos
            min_pos = tmp_pos
        else:
            if  np.random.uniform(0, 1) < np.exp(- (now_val - min_val) / (start_temp / float(i + 1))):
                now_pos = tmp_pos
        this_val = seg[now_pos]
        if this_val < ab_thresh:
            return min_pos
        elif this_val < 2 * ab_thresh:
            if np.random.uniform(0, 1) < np.exp( - tipping_cnt / 2):
                return min_pos
    return min_pos

@cython.boundscheck(False)
@cython.wraparound(False)
def invAnnealing(np.ndarray[np.float64_t, ndim=1] seg, int end, int max_iter = 20, float thresh = 0.5, float start_temp = 0.8):
    cdef int now_pos = end
    cdef int min_pos = now_pos
    cdef int tipping_cnt = 0
    cdef int i = 0
    cdef int step = 0
    cdef int tmp_pos = now_pos
    cdef float this_val = 0.0
    cdef float min_val = seg[now_pos]
    cdef float start_val = min_val
    for i in range(max_iter):
        if i > 6:
            step = np.random.choice((1, 2)) * np.random.choice((-1, 1, 1))
        elif i > 2:
            step = np.random.choice((1, 2, 3)) * np.random.choice((-1, 1, 1))
        else:
            step = np.random.choice((2, 3))
        tmp_pos = now_pos - step
        if tmp_pos < 0:
            tmp_pos = 0
        elif tmp_pos <= 0:
            tmp_pos = 0
        now_val = seg[tmp_pos]
        if now_val < min_val:       # 接受当前移动以及最终移动
            min_val = now_val
            now_pos = tmp_pos
            min_pos = tmp_pos
        else:
            if  np.random.uniform(0, 1) < np.exp(- (now_val - min_val) / (start_temp / float(i + 1))):
                now_pos = tmp_pos
        this_val = seg[now_pos]
        if this_val < 0.05:
            break
        elif this_val < 0.1:
            if np.random.uniform(0, 1) < np.exp( - tipping_cnt / 2):
                break
            tipping_cnt += 1
    if this_val < thresh * start_val:
        return min_pos, True
    return min_pos, False

@cython.boundscheck(False)
@cython.wraparound(False)
def calculateMeanAmp(np.ndarray[np.float64_t, ndim=2] y):
    cdef float wlen = float(y.shape[0])
    cdef int fn = y.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] amps = np.zeros(fn)
    cdef int i = 0
    for i in range(fn):
        amps[i] = sum(abs(y[:, i])) / wlen
    return amps

@cython.boundscheck(False)
@cython.wraparound(False)
def faultsFiltering(np.ndarray[np.float64_t, ndim=1] amps, starts, ends, float thresh):
    cdef int max_val = max(amps)
    _starts = []
    _ends = []
    if thresh > max_val / 4:
        thresh = max_val / 4
    for start, end in zip(starts, ends):
        if np.mean(amps[start:end]) > thresh and end - start >= 12: # 单门限阈值 长度限制(__min_ken__)
            _starts.append(start)
            _ends.append(end)
    return _starts, _ends

@cython.boundscheck(False)
@cython.wraparound(False)
def zeroCrossingRate(np.ndarray[np.float64_t, ndim = 2] y, int fn):
    cdef int i = 0
    cdef int j = 0
    cdef int wlen = y.shape[0] - 1
    cdef np.ndarray[np.float64_t, ndim=1] zcr = np.zeros(fn, dtype = np.float64)
    for i in range(fn):
        for j in range(wlen):
            if (y[j, i] > 0 and y[j + 1, i] < 0) or (y[j, i] < 0 and y[j + 1, i] > 0):
                zcr[i] += 1
    return zcr

@cython.boundscheck(False)
@cython.wraparound(False)
def findSegment(np.ndarray[np.int32_t, ndim=1] expr):
    cdef int i = 0
    cdef np.ndarray[np.int32_t, ndim=1] voice_index
    cdef int size = 0
    if expr[0] == 0:
        voice_index = np.arange(expr.size)[expr == 1]
    else:
        voice_index = expr
    size = voice_index.size
    starts = [voice_index[0]]
    ends = []
    for i in range(voice_index.size - 1):
        if voice_index[i + 1] - voice_index[i] > 1:
            starts.append(voice_index[i + 1])
            ends.append(voice_index[i])
    ends.append(voice_index[size - 1])
    return starts, ends

@cython.boundscheck(False)
@cython.wraparound(False)
def enframe(np.ndarray[float, ndim = 1] x, int win, int inc = 400):
    cdef int nx = len(x)
    cdef int nf = int(np.round((nx - win + inc) / inc))
    cdef int i = 0
    cdef np.ndarray[np.float64_t, ndim=2] f = np.zeros((win, nf), dtype = np.float64)
    for i in range(nf):
        temp = x[inc * i:inc * i + win]
        f[:temp.size, i] = x[inc * i:inc * i + win].ravel()      # 没有帧移
    return f

@cython.boundscheck(False)
# @cython.wraparound(False)
def vadSegment(y, zcrs, fn, NIS):
    cdef np.ndarray[np.float64_t, ndim=1] amp = sum(y ** 2)
    cdef float zcrth = np.mean(zcrs[:NIS])
    cdef float amp2 = 0.155                   # 如何进行鲁棒的噪音估计？
    cdef float amp1 = 0.205                   
    cdef float zcr2 = 0.15 * zcrth
    cdef int status  = 0
    cdef int xn = 0
    cdef int count = 0
    cdef int silence = 0
    cdef int n = 0
    cdef int e1 = 0
    cdef np.ndarray[np.int32_t, ndim=1] SF
    cdef np.ndarray[np.int32_t, ndim=1] NF
    cdef np.ndarray[np.int32_t, ndim=1] speechIndex
    x1 = [0]
    x2 = [0]
    for n in range(fn):
        if status <= 1:
            if amp[n] > amp1:
                x1[xn] = max(n - count - 1, 1)
                status = 2
                silence = 0
                count += 1
            elif amp[n] > amp2 or zcrs[n] < zcr2:
                status = 1
                count += 1
            else:
                status = 0
                count = 0
                x1[xn] = 0
                x2[xn] = 0
        elif status == 2:
            if amp[n] > amp2 or zcrs[n] < zcr2:
                count += 1
            else:
                silence += 1
                if silence < 10:
                    count += 1
                elif count < 12:
                    status = 0
                    silence = 0
                    count = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count
        else:       # status == 3
            status = 0
            xn += 1
            count = 0
            silence = 0
            x1.append(0)
            x2.append(0)
    e1 = len(x1)
    if x1[-1] == 0:
        e1 -= 1
    SF = np.zeros(fn, dtype = int)
    NF = np.ones(fn, dtype = int)
    for n in range(e1):
        SF[x1[n]:x2[n]] = 1
        NF[x1[n]:x2[n]] = 0
    speechIndex = np.arange(SF.size)[SF == 1]
    return findSegment(speechIndex)
```
