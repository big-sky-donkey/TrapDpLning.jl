mutable struct rnn
    wx
    wh
    b
    act
    state # state的尺寸==out==hide，由参数设置，称为隐藏层节点数
    function rnn(in, hide, act=tanh)
        wx=rand(hide,in)*0.02 .-0.01
        wh=rand(hide,hide)*0.02 .-0.01
        b=rand(hide)*0.02 .-0.01
        state=zeros(hide)
        new(wx,wh,b,act,state)
    end
end

function trainable(m::rnn)
    (m.wx, m.wh, m.b)
end

# 必须用comprehension算函子
function (m::rnn)(x)
    m.state = calc(x, m.state, m.wx, m.wh, m.b, m.act)
end
