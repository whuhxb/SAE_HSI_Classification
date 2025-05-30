function [params, netconfig] = stack2params(stack)

% Converts a "stack" structure into a flattened parameter vector and also
% stores the network configuration. This is useful when working with
% optimization toolboxes such as minFunc.
%将一个栈转换为一个扁平参数矢量，并且存储网络结构。当用最优化工具箱（如minFunc）工作的时候，这是有用的。
% [params, netconfig] = stack2params(stack)
%
% stack - the stack structure, where stack{1}.w = weights of first layer
%                                    第一层的权重
%                                    stack{1}.b = basis of first layer
%                                    第一层的偏差
%                                    stack{2}.w = weights of second layer
%                                    第二层的权重
%                                    stack{2}.b = basis of second layer
%                                    第二层的偏差
%                                    ... etc.


% Setup the compressed param vector 设置压缩参数向量
params = [];
for d = 1:numel(stack)
    %numel函数的作用是计算数组中元素个数，或者下标数组表达式的个数。
    % This can be optimized. But since our stacks are relatively short, it
    % is okay
    params = [params ; stack{d}.w(:) ; stack{d}.b(:) ]; %将栈中的元素作为向量进行输出
    
    % Check that stack is of the correct form 核对栈为正确的格式
    assert(size(stack{d}.w, 1) == size(stack{d}.b, 1), ...
        ['The bias should be a *column* vector of ' ...
         int2str(size(stack{d}.w, 1)) 'x1']);
    if d < numel(stack)
        assert(size(stack{d}.w, 1) == size(stack{d+1}.w, 2), ...
            ['The adjacent layers L' int2str(d) ' and L' int2str(d+1) ...
             ' should have matching sizes.']);
    end
    
end

if nargout > 1
    % Setup netconfig
    if numel(stack) == 0
        netconfig.inputsize = 0;%当栈为空的时候，inputsize的大小为0
        netconfig.layersizes = {};
    else
        netconfig.inputsize = size(stack{1}.w, 2); %netconfig的大小返回stack{1}.w的列数，当栈不为空的时候，inputsize的大小为w的列数
        netconfig.layersizes = {};
        for d = 1:numel(stack)
            netconfig.layersizes = [netconfig.layersizes ; size(stack{d}.w,1)];%根据stack的大小确定layersizes的大小
        end
    end
end

end