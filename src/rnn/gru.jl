# gru
# 对lstm的简化

# 不把act设为参数，直接使用sigmoid和tanh
mutable struct gru
    # 2个门ru的参数
    wrx
    wrh
    br
    wux
    wuh
    bu
    # 即时状态ct的参数
    wcx
    wch
    bc
    # 1个状态
    state # state的尺寸==out==hide，由参数设置，称为隐藏层节点数
    function gru(in, out)
        # 2个门与1个即时状态的9个wb参数的初始化
        wrx=(rand(out,in)*2 .-1)/100
        wrh=(rand(out,out)*2 .-1)/100
        br=zeros(out)
        wux=(rand(out,in)*2 .-1)/100
        wuh=(rand(out,out)*2 .-1)/100
        bu=zeros(out)
        wcx=(rand(out,in)*2 .-1)/100
        wch=(rand(out,out)*2 .-1)/100
        bc=zeros(out)
        # 1个状态的初始化
        state=zeros(out)
        new(wrx, wrh, br, wux, wuh, bu, wcx, wch, bc, state)
    end
end

function trainable(m::gru)
    (m.wrx, m.wrh, m.br, 
    m.wux, m.wuh, m.bu, 
    m.wcx, m.wch, m.bc)
end

# 必须用comprehension算函子
function (m::gru)(x)
    rg = calc(x, m.state, m.wrx, m.wrh, m.br, sigmoid)
    ug = calc(x, m.state, m.wux, m.wuh, m.bu, sigmoid)
    ct = calc(x, rg.*m.state, m.wcx, m.wch, m.bc, tanh)
    m.state=ug.*ct-(ug.-1).*m.state
end
