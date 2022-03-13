# 全连接层，用于把cnn得到的4维数据转为1维
struct fullconnect
    w
    b
    function fullconnect(in, out)
        w=rand(out, in).-0.5
        b=zeros(out)
        new(w, b)
    end
end

# 图片的第四个index分开调用函子，123个index在全连接层合并为vec
# 全连接层返回一个vec，需要求和或取值才能与数字y相减
function (m::fullconnect)(x)
    x=[x[i] for i in 1:prod(size(x))]
    m.w*x+m.b
end

function trainable(m::fullconnect)
    (m.w, m.b)
end
