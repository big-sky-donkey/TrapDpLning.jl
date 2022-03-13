# lstm
# 解决rnn在长序列中梯度消失爆炸的问题（多重嵌套的计算会导致梯度消失爆炸）
# 3个门控制遗忘、输入、输出，可以计算每个单元的状态c和神经网络的状态state

# 不把act设为参数，直接使用sigmoid和tanh
mutable struct lstm
    # 3个门fio的参数
    wfx
    wfh
    bf
    wix
    wih
    bi
    wox
    woh
    bo
    # 即时状态ct的参数
    wcx
    wch
    bc
    # 2个状态
    c
    state # state的尺寸==out==hide，由参数设置，称为隐藏层节点数
    # 6个各个时刻的向量列表，不作为字段，直接由前一个计算后一个(6个: c~ch fio)

    function lstm(in, out)
        # 3个门与1个即时状态的12个wb参数的初始化
        wfx=(rand(out,in)*2 .-1)/100
        wfh=(rand(out,out)*2 .-1)/100
        bf=zeros(out)
        wix=(rand(out,in)*2 .-1)/100
        wih=(rand(out,out)*2 .-1)/100
        bi=zeros(out)
        wox=(rand(out,in)*2 .-1)/100
        woh=(rand(out,out)*2 .-1)/100
        bo=zeros(out)
        wcx=(rand(out,in)*2 .-1)/100
        wch=(rand(out,out)*2 .-1)/100
        bc=zeros(out)
        # 2个状态的初始化
        c=zeros(out)
        state=zeros(out)
        new(wfx, wfh, bf, wix, wih, bi, wox, woh, bo, wcx, wch, bc, c, state)
    end
end

function trainable(m::lstm)
    (m.wfx, m.wfh, m.bf, 
    m.wix, m.wih, m.bi, 
    m.wox, m.woh, m.bo, 
    m.wcx, m.wch, m.bc)
end

# 必须用comprehension算函子
function (m::lstm)(x)
    # 3个门的计算
    fg = calc(x, m.state, m.wfx, m.wfh, m.bf, sigmoid)
    ig = calc(x, m.state, m.wix, m.wih, m.bi, sigmoid)
    og = calc(x, m.state, m.wox, m.woh, m.bo, sigmoid)
    # 即时状态
    ct = calc(x, m.state, m.wcx, m.wch, m.bc, tanh)
    # 单元状态
    m.c = fg.*m.c+ig.*ct
    # 状态
    m.state = og.*(tanh.(m.c))
end
