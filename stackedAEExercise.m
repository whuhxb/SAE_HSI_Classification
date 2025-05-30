%% CS294A/CS294W Stacked Autoencoder Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  stacked autoencoder exercise. You will need to complete code in
%  stackedAECost.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises. You will need the initializeParameters.m
%  loadMNISTImages.m, and loadMNISTLabels.m files from previous exercises.
%  
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below. 提供相关的参数值使sparse
%  autoencoder可以得到好的滤波，并且不需要改变下面参数的值。
inputSize = 10*10;
numClasses =9;
hiddenSizeL1 = 800;    % Layer 1 Hidden Size
% hiddenSizeL2 = 25;    % Layer 2 Hidden Size
% sP=[0.0001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
sparsityParam = 0.0001;   % desired average activation of the hidden units. 隐藏单元的平均激活值
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-6;         % weight decay parameter 权重衰减参数      
beta = 0.001;              % weight of sparsity penalty term  稀疏惩罚项权重     

%%======================================================================
%% STEP 1: Load data from the MNIST database
% 从MNIST数据集中下载数据
%  This loads our training data from the MNIST database files.

% %LoadData from the folder
% data=loadData();
% ratho=0.5;
% [~,c]=size(data);

%读入训练样本区辐射亮度值矩阵
f=xlsread('trainroi100.xls');
[np,mp]=size(f);%此时np的值为2832，mp的值为144
numImages=np;
numRows=10;
numCols=mp/numRows;%即numCols的值为12
max_band=[];
min_band=[];
imagein=[];
for i=1:mp
    max_band(i)=max(f(:,i));
    min_band(i)=min(f(:,i));
    imagein(:,i)=(f(:,i)-min_band(i))/(max_band(i)-min_band(i));%对影像的输入数据进行归一化
end
%这里是针对每一个波段进行最大最小值归一化
%下面表示对每个类别进行标记
% trainData=[];
trainData=imagein;
trainData=trainData';%此时trainData的大小为200行4904列
% trainData=reshape(imagein,numCols,numRows,numImages);%导入的数据的顺序还有些问题，需要思考一下？？
% trainData=permute(trainData,[2 1 3]);
% trainData=reshape(trainData,size(trainData,1)*size(trainData,2),size(trainData,3));%其实变换了这么多，感觉就直接可以用trainData=imagein来表示。
% trainLables=[];
m1=100;
m2=100;
m3=100;
m4=100;
m5=100;
m6=100;
m7=100;
m8=100;
m9=100;
trainLabels=[ones(m1,1);2*ones(m2,1);3*ones(m3,1);4*ones(m4,1);5*ones(m5,1);6*ones(m6,1);7*ones(m7,1);...
    8*ones(m8,1);9*ones(m9,1)];
%trainLabels是用列向量表示的
%trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

%%======================================================================
%% STEP 2: Train the first sparse autoencoder 训练第一个稀疏自编码器
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.在未标记的SIL训练影像中训练第一个稀疏自编码器。
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here. 如果你已经正确使用过sparseAutoencoderCost.m，那么这里就不必改变什么了。


%  Randomly initialize the parameters 随机初始化参数
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);%hiddenSizeL1大小为60，inputSize大小为200，用这两个值去初始化W1
%initializeParameters的作用在这里是分别对隐藏层的权重和偏差矩阵进行初始化，sae1Theta中包含着W1，W2，b1,b2
%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                训练第一层稀疏自编码器，这一层有一个hiddenSizeL1大小的隐层
%                You should store the optimal parameters in sae1OptTheta
%                将最优参数保存在sae1OptTheta中
addpath minFunc/;%将addpath后面的内容作为搜索路径
options = struct;%定义一个结构数组
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';
sae1OptTheta =  minFunc(@(p)sparseAutoencoderCost(p,...
    inputSize,hiddenSizeL1,lambda,sparsityParam,beta,trainData),sae1Theta,options);%训练出第一层网络的参数
%sae1OptTheta中保存的是W1grad,W2grad，b1grad,b2grad的值，即后向传播的权重改变值


% -------------------------------------------------------------------------



%%======================================================================
%% STEP 2: Train the second sparse autoencoder 训练第二个稀疏自编码器
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse. 在第一个自编码特征的基础上训练第二个稀疏自编码器。
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);
%  feedForwardAutoencoder函数的作用是将输入的原始数据经过第一步稀疏编码后，得到激活值，然后将该激活值输入到第二个稀疏自编码器
%  中，其中前向传播的时候W1，b1的大小是经过了梯度调整的

%  Randomly initialize the parameters
%  训练第二个稀疏自编码器的时候，权重值W1，W2，b1,b2是重新随机初始化的
% sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);%将hiddenSizeL2值为30，hiddenSizeL1的值为60,将其分别初始化为权重W2的大小
% 
% %% ---------------------- YOUR CODE HERE  ---------------------------------
% %  Instructions: Train the second layer sparse autoencoder, this layer has
% %                an hidden size of "hiddenSizeL2" and an inputsize of
% %                "hiddenSizeL1"
% %
% %                You should store the optimal parameters in sae2OptTheta
% %这一过程与训练第一个稀疏自编码器的函数用法是不太一样的
% 
% sae2OptTheta =  minFunc(@(p)sparseAutoencoderCost(p,...
%     hiddenSizeL1,hiddenSizeL2,lambda,sparsityParam,beta,sae1Features),sae2Theta,options);%训练出第一层网络的参数
% 


% -------------------------------------------------------------------------



%%======================================================================
%% STEP 3: Train the softmax classifier 训练softmax分类器
%  This trains the sparse autoencoder on the second autoencoder features.
%  在第二个自编码特征中训练稀疏自编码。
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.

% [sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, sae1Features);
% %这里将经过第二个稀疏编码后得到的值作为特征输入到softmax分类器中
%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL1 * numClasses, 1);%saeSoftmaxTheta的大小是随机初始化的，类似于之前的W1，W2，b1,b2组成的一个行向量


%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);

softmaxLambda = 1e-4; %softmaxLambda参数在这里有什么含义呢？
softoptions.maxIter = 100;
softmaxModel = softmaxTrain(hiddenSizeL1,numClasses,softmaxLambda,...
                            sae1Features,trainLabels,softoptions);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);%softmaxModel.optTheta中存储的是

% -------------------------------------------------------------------------


%%======================================================================
%% STEP 5: Finetune softmax model 调整softmax模型
%利用2个隐含层和一个softmax分类器所有的参数对整个系统进行微调
% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned 利用已学习到的参数初始化这个栈
stack = cell(1,1);%创建一个两行一列的单元型变量stack,分别用来存储第一次稀疏自编码的W1，b1和第二次稀疏自编码的W2，b2
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
% stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
%                      hiddenSizeL2, hiddenSizeL1);
% stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model 初始化deep model的参数
[stackparams, netconfig] = stack2params(stack);%调用stack2params，将stack转换为一个参数向量，用于存储权重结构
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];%stackedAETheta中同时包括saeSoftmaxOptTheta和stackparams（即每层的权重和偏差及输入数据的大小、层数）
%stackedAETheta是个向量，为整个网络的参数，包括分类器那部分，且分类器那部分的参数放前面
%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2". 
%  隐藏单元的大小这里是指输入到分类器的维数，其余hiddenSizeL2相对应。
options.maxIter = 400;
[stackedAEOptTheta, cost] =  minFunc(@(p)stackedAECost(p,inputSize,hiddenSizeL1,...
                         numClasses, netconfig,lambda, trainData, trainLabels),...
                        stackedAETheta,options);%训练出第一层网络的参数，stcakedAECost函数返回的是cost和gradient的值

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
% testData = loadMNISTImages('t10k-images.idx3-ubyte');
% testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
f1=xlsread('testroi500.xls');
[nq,mq]=size(f1);
numImages1=nq;
numRows1=10;
numCols1=mq/numRows1;%即Cols的值为12
max_band1=[];
min_band1=[];
imagein1=[];
for i=1:mq
    max_band1(i)=max(f1(:,i));
    min_band1(i)=min(f1(:,i));
    imagein1(:,i)=(f1(:,i)-min_band1(i))/(max_band1(i)-min_band1(i));%对影像的输入数据进行归一化
end

% testData=[];
% testData=reshape(imagein1,numCols1,numRows1,numImages1);%导入的数据的顺序还有些问题，需要思考一下？？
% testData=permute(testData,[2 1 3]);
% testData=reshape(testData,size(testData,1)*size(testData,2),size(testData,3));%其实变换了这么多，感觉就直接可以用trainData=imagein来表示。
testData=imagein1;
testData=testData'
%下面表示对每个类别进行标记
% testLables=[];
ml1=500;
ml2=500;
ml3=500;
ml4=500;
ml5=500;
ml6=500;
ml7=500;
ml8=500;
ml9=500;
testLabels=[ones(ml1,1);2*ones(ml2,1);3*ones(ml3,1);4*ones(ml4,1);5*ones(ml5,1);6*ones(ml6,1);7*ones(ml7,1);...
    8*ones(ml8,1);9*ones(ml9,1)];
%stackedAEPredict函数用来评价分类精度
[pred1] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL1, ...
                          numClasses, netconfig, testData);
%stackedAETheta是没有经过变换的权重和偏差的矩阵
acc1 = mean(testLabels(:) == pred1(:));%acc得到的输入的类别标记和输出类别标记的平均值，自己还不是特别懂精度为什么要这样求解？？
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc1 * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL1, ...
                          numClasses, netconfig, testData);
%stackedAEOptTheta是经过梯度调整后的权重和偏差组成的一个矩阵
acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images 准确率是正确分类影像的比例
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
