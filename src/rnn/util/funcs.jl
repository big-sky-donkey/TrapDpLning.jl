function sigmoid(x)
    1.0 / (1.0 + exp(-1.0 * x))
end
# tanh()可以直接用
# function tanh(x)
#     2.0 / (1.0 + exp(-2.0 * x)) - 1.0
# end

# 通用函数calc，提取出了在多个rnn中公共的计算
function calc(x, h, wx, wh, b, act=tanh)
    act.(wx*x + wh*h + b)
end

# 3个rnn
# 公式上，这里是，算出h直接作为结果y，在使用时常常要在后面chain一个全连接层，否则隐藏层节点数==输出数
# 另一种可以考虑的选择是，先计算出h，再用h套另一组actwb算出y（也就是自带了一个感知机）

# birnn
# 可以组合任意两个rnn，一个正向遍历序列，一个反向遍历序列
