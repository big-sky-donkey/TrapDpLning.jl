# birnn
# 可以组合任意两个rnn，一个正向遍历序列，一个反向遍历序列
# 双拼：封装两个rnn，参数独立，对于某个序列，两个rnn分别正向和反向遍历序列，计算出两个state，求和作为结果
# 实现：平常函子都是一个参数，birnn函子设置两个参数，函子计算出两个rnn调函子之和，调用时输入一正序一反序即可

struct birnn
    rnn1
    rnn2
    function birnn(rnn1, rnn2) # 参数：rnn or lstm or gru
        new(rnn1, rnn2)
    end
end

function trainable(m::birnn)
    (m.rnn1, m.rnn2)
end

function (m::birnn)(x1, x2)
    rnn1, rnn2 = m.rnn1, m.rnn2
    rnn1(x1) + rnn2(x2)
end
