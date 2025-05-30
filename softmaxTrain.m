function [softmaxModel] = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options)
%softmaxTrain Train a softmax model with the given parameters on the given
% data. Returns softmaxOptTheta, a vector containing the trained parameters
% for the model.
%softmaxTrain在给定参数下训练一个softmax模型，并且返回softmaxOptTheta，其中softmaxOptTheta是一个包含
%训练参数的模型。
% inputSize: the size of an input vector x^(i)输入矢量的大小，这里面即hiddenSizeL2的大小
% numClasses: the number of classes 类别个数
% lambda: weight decay parameter 权重衰减参数
% inputData: an N by M matrix containing the input data, such that
%            inputData(:, c) is the cth input输入数据矩阵
% labels: M by 1 matrix containing the class labels for the
%            corresponding inputs. labels(c) is the class label for
%            the cth input 对于每个输入的类别标记
% options (optional): options
%   options.maxIter: number of iterations to train for 训练的迭代次数

if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 400;
end

% initialize parameters
theta = 0.005 * randn(numClasses * inputSize, 1);

% Use minFunc to minimize the function
addpath minFunc/;
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost 用L-BFGS去最优化我们的cost function.
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our
                          % problem, 在下面的function中，我们需要两个输出，即方程值和梯度
                          % softmaxCost.m satisfies this.
minFuncOptions.display = 'on';

[softmaxOptTheta, cost] = minFunc( @(p) softmaxCost(p, ...
                                   numClasses, inputSize, lambda, ...
                                   inputData, labels), ...                                   
                              theta, options);%一直不是很清楚p在这里到底是什么意思？

% Fold softmaxOptTheta into a nicer format
softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);%optTheta值中其实包含的是softmaxOptTheta的值，经reshape转换之后为numClasses行inputSize列
softmaxModel.inputSize = inputSize;%inputSize的大小为30
softmaxModel.numClasses = numClasses;%numClasses的大小为10
                          
end                          
