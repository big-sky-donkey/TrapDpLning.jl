# 把多个神经网络层串联起来
struct chain
    layers
end

function trainable(m::chain)
    m.layers
end

function (m::chain)(xs...)     # 接收多个参数的情况，只用于birnn
    res=xs
    for layer in m.layers
        if typeof(res)<:Tuple
            res=layer(res...)  # 将tuple展开为多个参数，只用于birnn
        else
            res=layer(res)
        end
    end
    res
end

# 需要训练的参数，返回所有层
# 对于ary|vec|mat，会收集参数，对于其他，会递归寻找ary|vec|mat
function chain(layers...)
    chain(layers)
end
