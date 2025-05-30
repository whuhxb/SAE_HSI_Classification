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
%  change the parameters below. �ṩ��صĲ���ֵʹsparse
%  autoencoder���Եõ��õ��˲������Ҳ���Ҫ�ı����������ֵ��
inputSize = 10*10;
numClasses =9;
hiddenSizeL1 = 800;    % Layer 1 Hidden Size
% hiddenSizeL2 = 25;    % Layer 2 Hidden Size
% sP=[0.0001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
sparsityParam = 0.0001;   % desired average activation of the hidden units. ���ص�Ԫ��ƽ������ֵ
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-6;         % weight decay parameter Ȩ��˥������      
beta = 0.001;              % weight of sparsity penalty term  ϡ��ͷ���Ȩ��     

%%======================================================================
%% STEP 1: Load data from the MNIST database
% ��MNIST���ݼ�����������
%  This loads our training data from the MNIST database files.

% %LoadData from the folder
% data=loadData();
% ratho=0.5;
% [~,c]=size(data);

%����ѵ����������������ֵ����
f=xlsread('trainroi100.xls');
[np,mp]=size(f);%��ʱnp��ֵΪ2832��mp��ֵΪ144
numImages=np;
numRows=10;
numCols=mp/numRows;%��numCols��ֵΪ12
max_band=[];
min_band=[];
imagein=[];
for i=1:mp
    max_band(i)=max(f(:,i));
    min_band(i)=min(f(:,i));
    imagein(:,i)=(f(:,i)-min_band(i))/(max_band(i)-min_band(i));%��Ӱ����������ݽ��й�һ��
end
%���������ÿһ�����ν��������Сֵ��һ��
%�����ʾ��ÿ�������б��
% trainData=[];
trainData=imagein;
trainData=trainData';%��ʱtrainData�Ĵ�СΪ200��4904��
% trainData=reshape(imagein,numCols,numRows,numImages);%��������ݵ�˳����Щ���⣬��Ҫ˼��һ�£���
% trainData=permute(trainData,[2 1 3]);
% trainData=reshape(trainData,size(trainData,1)*size(trainData,2),size(trainData,3));%��ʵ�任����ô�࣬�о���ֱ�ӿ�����trainData=imagein����ʾ��
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
%trainLabels������������ʾ��
%trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

%%======================================================================
%% STEP 2: Train the first sparse autoencoder ѵ����һ��ϡ���Ա�����
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.��δ��ǵ�SILѵ��Ӱ����ѵ����һ��ϡ���Ա�������
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here. ������Ѿ���ȷʹ�ù�sparseAutoencoderCost.m����ô����Ͳ��ظı�ʲô�ˡ�


%  Randomly initialize the parameters �����ʼ������
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);%hiddenSizeL1��СΪ60��inputSize��СΪ200����������ֵȥ��ʼ��W1
%initializeParameters�������������Ƿֱ�����ز��Ȩ�غ�ƫ�������г�ʼ����sae1Theta�а�����W1��W2��b1,b2
%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                ѵ����һ��ϡ���Ա���������һ����һ��hiddenSizeL1��С������
%                You should store the optimal parameters in sae1OptTheta
%                �����Ų���������sae1OptTheta��
addpath minFunc/;%��addpath�����������Ϊ����·��
options = struct;%����һ���ṹ����
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';
sae1OptTheta =  minFunc(@(p)sparseAutoencoderCost(p,...
    inputSize,hiddenSizeL1,lambda,sparsityParam,beta,trainData),sae1Theta,options);%ѵ������һ������Ĳ���
%sae1OptTheta�б������W1grad,W2grad��b1grad,b2grad��ֵ�������򴫲���Ȩ�ظı�ֵ


% -------------------------------------------------------------------------



%%======================================================================
%% STEP 2: Train the second sparse autoencoder ѵ���ڶ���ϡ���Ա�����
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse. �ڵ�һ���Ա��������Ļ�����ѵ���ڶ���ϡ���Ա�������
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);
%  feedForwardAutoencoder�����������ǽ������ԭʼ���ݾ�����һ��ϡ�����󣬵õ�����ֵ��Ȼ�󽫸ü���ֵ���뵽�ڶ���ϡ���Ա�����
%  �У�����ǰ�򴫲���ʱ��W1��b1�Ĵ�С�Ǿ������ݶȵ�����

%  Randomly initialize the parameters
%  ѵ���ڶ���ϡ���Ա�������ʱ��Ȩ��ֵW1��W2��b1,b2�����������ʼ����
% sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);%��hiddenSizeL2ֵΪ30��hiddenSizeL1��ֵΪ60,����ֱ��ʼ��ΪȨ��W2�Ĵ�С
% 
% %% ---------------------- YOUR CODE HERE  ---------------------------------
% %  Instructions: Train the second layer sparse autoencoder, this layer has
% %                an hidden size of "hiddenSizeL2" and an inputsize of
% %                "hiddenSizeL1"
% %
% %                You should store the optimal parameters in sae2OptTheta
% %��һ������ѵ����һ��ϡ���Ա������ĺ����÷��ǲ�̫һ����
% 
% sae2OptTheta =  minFunc(@(p)sparseAutoencoderCost(p,...
%     hiddenSizeL1,hiddenSizeL2,lambda,sparsityParam,beta,sae1Features),sae2Theta,options);%ѵ������һ������Ĳ���
% 


% -------------------------------------------------------------------------



%%======================================================================
%% STEP 3: Train the softmax classifier ѵ��softmax������
%  This trains the sparse autoencoder on the second autoencoder features.
%  �ڵڶ����Ա���������ѵ��ϡ���Ա��롣
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.

% [sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, sae1Features);
% %���ｫ�����ڶ���ϡ������õ���ֵ��Ϊ�������뵽softmax��������
%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL1 * numClasses, 1);%saeSoftmaxTheta�Ĵ�С�������ʼ���ģ�������֮ǰ��W1��W2��b1,b2��ɵ�һ��������


%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);

softmaxLambda = 1e-4; %softmaxLambda������������ʲô�����أ�
softoptions.maxIter = 100;
softmaxModel = softmaxTrain(hiddenSizeL1,numClasses,softmaxLambda,...
                            sae1Features,trainLabels,softoptions);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);%softmaxModel.optTheta�д洢����

% -------------------------------------------------------------------------


%%======================================================================
%% STEP 5: Finetune softmax model ����softmaxģ��
%����2���������һ��softmax���������еĲ���������ϵͳ����΢��
% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned ������ѧϰ���Ĳ�����ʼ�����ջ
stack = cell(1,1);%����һ������һ�еĵ�Ԫ�ͱ���stack,�ֱ������洢��һ��ϡ���Ա����W1��b1�͵ڶ���ϡ���Ա����W2��b2
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
% stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
%                      hiddenSizeL2, hiddenSizeL1);
% stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model ��ʼ��deep model�Ĳ���
[stackparams, netconfig] = stack2params(stack);%����stack2params����stackת��Ϊһ���������������ڴ洢Ȩ�ؽṹ
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];%stackedAETheta��ͬʱ����saeSoftmaxOptTheta��stackparams����ÿ���Ȩ�غ�ƫ��������ݵĴ�С��������
%stackedAETheta�Ǹ�������Ϊ��������Ĳ����������������ǲ��֣��ҷ������ǲ��ֵĲ�����ǰ��
%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2". 
%  ���ص�Ԫ�Ĵ�С������ָ���뵽��������ά��������hiddenSizeL2���Ӧ��
options.maxIter = 400;
[stackedAEOptTheta, cost] =  minFunc(@(p)stackedAECost(p,inputSize,hiddenSizeL1,...
                         numClasses, netconfig,lambda, trainData, trainLabels),...
                        stackedAETheta,options);%ѵ������һ������Ĳ�����stcakedAECost�������ص���cost��gradient��ֵ

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
numCols1=mq/numRows1;%��Cols��ֵΪ12
max_band1=[];
min_band1=[];
imagein1=[];
for i=1:mq
    max_band1(i)=max(f1(:,i));
    min_band1(i)=min(f1(:,i));
    imagein1(:,i)=(f1(:,i)-min_band1(i))/(max_band1(i)-min_band1(i));%��Ӱ����������ݽ��й�һ��
end

% testData=[];
% testData=reshape(imagein1,numCols1,numRows1,numImages1);%��������ݵ�˳����Щ���⣬��Ҫ˼��һ�£���
% testData=permute(testData,[2 1 3]);
% testData=reshape(testData,size(testData,1)*size(testData,2),size(testData,3));%��ʵ�任����ô�࣬�о���ֱ�ӿ�����trainData=imagein����ʾ��
testData=imagein1;
testData=testData'
%�����ʾ��ÿ�������б��
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
%stackedAEPredict�����������۷��ྫ��
[pred1] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL1, ...
                          numClasses, netconfig, testData);
%stackedAETheta��û�о����任��Ȩ�غ�ƫ��ľ���
acc1 = mean(testLabels(:) == pred1(:));%acc�õ������������Ǻ��������ǵ�ƽ��ֵ���Լ��������ر𶮾���ΪʲôҪ������⣿��
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc1 * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL1, ...
                          numClasses, netconfig, testData);
%stackedAEOptTheta�Ǿ����ݶȵ������Ȩ�غ�ƫ����ɵ�һ������
acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images ׼ȷ������ȷ����Ӱ��ı���
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
