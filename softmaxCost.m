function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);%表示data的第二维元素

groundTruth = full(sparse(labels, 1:numCases, 1));%full表示将稀疏矩阵转换为满矩阵，sparse表示讲一个满矩阵转换为一个稀疏矩阵
cost = 0;

thetagrad = zeros(numClasses, inputSize);
%thetagrad矩阵为numClasses行，inputSize列
%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
% M = bsxfun(@minus,theta*data,max(theta*data, [], 1));%@minus表示二进制的减，theta*data表示矩阵
% M = exp(M);%对M矩阵求指数
% p = bsxfun(@rdivide, M, sum(M));%bsxfun函数的作用是二进制单例扩张函数，@rdivide表示数组右除
% cost = -1/numCases * groundTruth(:)' * log(p(:)) + lambda/2 * sum(theta(:) .^ 2);%sum(theta(:).^2)表示对theta矩阵所有元素求和
% thetagrad = -1/numCases * (groundTruth - p) * data' + lambda * theta;%softmax中的梯度函数
%上面的代码是群主博客中复制过来的，下面的是Wilbur自己写的
probs = theta * data;  % numClasses x numCases
probs = exp(bsxfun(@minus, probs, max(probs, [], 1)));
probs = bsxfun(@rdivide, probs, sum(probs));
n = size(data, 2);
cost = -sum(log(probs(logical(groundTruth)))) / n + lambda/2 * sum(sum(theta.^2));
thetagrad = -(groundTruth - probs) * data' / n + lambda * theta;
% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

