function theta = initializeParameters(hiddenSize, visibleSize) %参数的初始化大小为隐藏单元大小hiddenSize和可见单元大小visibleSize

%% Initialize parameters randomly based on layer sizes.基于每层的大小随机初始化参数
r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r] 我们一律从[-r,r]中选择权重大小
W1 = rand(hiddenSize, visibleSize) * 2 * r - r;  %W1的大小为hiddenSize行，visibleSize列，在随机初始化矩阵hiddenSize行visibleSize列的基础上，分别乘以2*r,然后总体在减去r
W2 = rand(visibleSize, hiddenSize) * 2 * r - r;  %W2的大小为visibleSize行，hiddenSize列
%W1和W2中r的出现是为了确定权重随机值的大小的
b1 = zeros(hiddenSize, 1); %b1的大小为hiddenSize行1列
b2 = zeros(visibleSize, 1); %b2的大小为visibleSize行1列

% Convert weights and bias gradients to the vector form. 将权重和偏差梯度转换为矢量形式
% This step will "unroll" (flatten and concatenate together) all
% 这一步将会将所有的参数展开为一个矢量（展开过程包括平整和连接在一起），然后改矢量可以被用到minFunc中，但是具体会怎么用呢？？
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end
