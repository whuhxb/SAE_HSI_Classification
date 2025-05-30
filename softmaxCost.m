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

numCases = size(data, 2);%��ʾdata�ĵڶ�άԪ��

groundTruth = full(sparse(labels, 1:numCases, 1));%full��ʾ��ϡ�����ת��Ϊ������sparse��ʾ��һ��������ת��Ϊһ��ϡ�����
cost = 0;

thetagrad = zeros(numClasses, inputSize);
%thetagrad����ΪnumClasses�У�inputSize��
%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
% M = bsxfun(@minus,theta*data,max(theta*data, [], 1));%@minus��ʾ�����Ƶļ���theta*data��ʾ����
% M = exp(M);%��M������ָ��
% p = bsxfun(@rdivide, M, sum(M));%bsxfun�����������Ƕ����Ƶ������ź�����@rdivide��ʾ�����ҳ�
% cost = -1/numCases * groundTruth(:)' * log(p(:)) + lambda/2 * sum(theta(:) .^ 2);%sum(theta(:).^2)��ʾ��theta��������Ԫ�����
% thetagrad = -1/numCases * (groundTruth - p) * data' + lambda * theta;%softmax�е��ݶȺ���
%����Ĵ�����Ⱥ�������и��ƹ����ģ��������Wilbur�Լ�д��
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

