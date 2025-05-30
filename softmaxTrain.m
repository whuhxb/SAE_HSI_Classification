function [softmaxModel] = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options)
%softmaxTrain Train a softmax model with the given parameters on the given
% data. Returns softmaxOptTheta, a vector containing the trained parameters
% for the model.
%softmaxTrain�ڸ���������ѵ��һ��softmaxģ�ͣ����ҷ���softmaxOptTheta������softmaxOptTheta��һ������
%ѵ��������ģ�͡�
% inputSize: the size of an input vector x^(i)����ʸ���Ĵ�С�������漴hiddenSizeL2�Ĵ�С
% numClasses: the number of classes ������
% lambda: weight decay parameter Ȩ��˥������
% inputData: an N by M matrix containing the input data, such that
%            inputData(:, c) is the cth input�������ݾ���
% labels: M by 1 matrix containing the class labels for the
%            corresponding inputs. labels(c) is the class label for
%            the cth input ����ÿ������������
% options (optional): options
%   options.maxIter: number of iterations to train for ѵ���ĵ�������

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
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost ��L-BFGSȥ���Ż����ǵ�cost function.
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our
                          % problem, �������function�У�������Ҫ���������������ֵ���ݶ�
                          % softmaxCost.m satisfies this.
minFuncOptions.display = 'on';

[softmaxOptTheta, cost] = minFunc( @(p) softmaxCost(p, ...
                                   numClasses, inputSize, lambda, ...
                                   inputData, labels), ...                                   
                              theta, options);%һֱ���Ǻ����p�����ﵽ����ʲô��˼��

% Fold softmaxOptTheta into a nicer format
softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);%optThetaֵ����ʵ��������softmaxOptTheta��ֵ����reshapeת��֮��ΪnumClasses��inputSize��
softmaxModel.inputSize = inputSize;%inputSize�Ĵ�СΪ30
softmaxModel.numClasses = numClasses;%numClasses�Ĵ�СΪ10
                          
end                          
