abstract type ty_AbstractOptimiser end

# sgd系列
# sgd
mutable struct sgd <: ty_AbstractOptimiser
    lr
    function sgd(lr=0.1)
        new(lr)
    end
end

# sgdm: Momentun, 带动量
mutable struct sgdm <: ty_AbstractOptimiser
    lr
    beta
    m_pre # 前一次的动量，初始化为0
    function sgdm(lr=0.1, beta=0.9)
        new(lr, beta, -1)
    end
end

# sgda: Nesterov Accelerated Gradient(NAG), 带加速
mutable struct sgda <: ty_AbstractOptimiser
    lr
    beta
    m_pre # 前一次的动量，初始化为0
    function sgda(lr=0.1, beta=0.9)
        new(lr, beta, -1)
    end
end

# adapt系列
# adagrad: 这个算法，r太大，走不动，经常是还没到目标就平缓了
mutable struct adagrad <: ty_AbstractOptimiser
    lr
    r  # 累计平方梯度，也称为二阶动量
    function adagrad(lr=0.1)
        new(lr, 0)
    end
end

# adadelta: rmsprop
mutable struct adadelta <: ty_AbstractOptimiser
    lr
    decay
    r
    function adadelta(lr=0.1, decay=0.9)
        new(lr, decay, 0)
    end
end


# 合体系列
# adam: adapt + Momentun
mutable struct adam <: ty_AbstractOptimiser
    lr
    beta
    decay
    m_pre
    r
    function adam(lr=0.01, belta=0.9, decay=0.9)
        new(lr, belta, decay, -1, 0)
    end
end

# nadam: Nesterov + adapt + Momentun
mutable struct nadam <: ty_AbstractOptimiser
    lr
    beta
    decay
    m_pre
    r
    function nadam(lr=0.01, belta=0.9, decay=0.9)
        new(lr, belta, decay, -1, 0)
    end
end
