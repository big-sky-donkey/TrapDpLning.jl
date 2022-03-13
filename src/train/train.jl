# train!
# 拆data，调loss，求loss对ps导数，更新ps

# sgd
# 随机梯度下降
# ps -= lr * gt
# 缺点：下降慢，可能在局部最优点震荡后收敛
function train!(m, data, loss, opt::sgd)
    ps = params(m)
    for d in data
        gs=gradient(()->loss(d...), ps)
        for p in ps
            p .-= opt.lr*gs[p] # 必须广播或[]，否则是对对象p修改，广播或[]了才是对真实元素修改
        end
    end
end

# sgdm
# 随机梯度下降，带动量
# ps -= lr * (  belta*dt-1 + (1- belta)*gt )，dt-1表示上次的下降
# 优点：利用惯性越过局部最优点
function train!(m, data, loss, opt::sgdm)
    ps = params(m)
    for d in data
        gs=gradient(()->loss(d...), ps)
        gs_vec = typeof(gs[ps[1]][1])[]
        for p in ps
            for xi in gs[p]
                push!(gs_vec, xi)
            end
        end
        if opt.m_pre==-1                    # 上次的动量，第一次调用时初始化为0
            opt.m_pre=zeros(size(gs_vec,1))
        end

        cnt=0
        for p in ps
            for i in 1:prod(size(p))
                cnt=cnt+1
                p[i] = p[i] - opt.lr* ( opt.beta*opt.m_pre[cnt] + (1-opt.beta)*gs_vec[cnt] )
            end
        end
        opt.m_pre=deepcopy(gs_vec) # 不必deepcopy
    end
end

# sgda, also Nestero
# 随机梯度下降，带加速
# ps -= lr * (  belta*dt-1 )
# 求gt
# ps -= lr * (  belta*dt-1 + (1- belta)*gt )
# 优点：每次先让参数按上次往前走一步，收敛快，推导上是二阶导数所以叫加速
function train!(m, data, loss, opt::sgda)
    ps = params(m)
    for d in data
        if !(opt.m_pre==-1)
            cnt=0
            for p in ps
                for i in 1:prod(size(p))
                    cnt=cnt+1
                    p[i] = p[i] - opt.lr* opt.beta * opt.m_pre[cnt]
                end
            end
        end
        gs=gradient(()->loss(d...), ps)
        m_cur = typeof(gs[ps[1]][1])[]
        for p in ps
            for xi in gs[p]
                push!(m_cur, xi)
            end
        end
        if opt.m_pre==-1                   # 上次的动量，第一次调用时初始化为0
            opt.m_pre=zeros(size(m_cur,1))
        end

        cnt=0
        for p in ps
            for i in 1:prod(size(p))
                cnt=cnt+1
                p[i] = p[i] - opt.lr* ( opt.beta*opt.m_pre[cnt] + (1-opt.beta)*m_cur[cnt] )
            end
        end
        opt.m_pre=deepcopy(m_cur)
    end
end

# adagrad
# 自适应学习率梯度下降
# r = r + gt' * gt
# ps -= lr / ( eps+sqrt(r) ) * gt
# 优点：前面平缓了就多进步，前面陡峭了就少进步
# 缺点：r太大了，走不动
function train!(m, data, loss, opt::adagrad)
    ps = params(m)
    for d in data
        gs=gradient(()->loss(d...), ps)
        gs_vec = typeof(gs[ps[1]][1])[]
        for p in ps
            for xi in gs[p]
                push!(gs_vec, xi)
            end
        end
        opt.r = opt.r + sum(gs_vec.*gs_vec)
        cnt=0
        for p in ps
            for i in 1:prod(size(p))
                cnt=cnt+1
                p[i] = p[i] - opt.lr / ( (typeof(gs[ps[1]][1]))+sqrt(opt.r) ) * gs_vec[cnt]
            end
        end
    end
end

# adadelta, also rmsprop
# 自适应学习率梯度下降，带衰减
# r = decay*r + (1-decay) * (gt'*gt)
# ps -= lr / ( eps+sqrt(r) ) * gt
# 优点：给r加了个衰减系数decay，避免r太大走不动
function train!(m, data, loss, opt::adadelta)
    ps = params(m)
    for d in data
        gs=gradient(()->loss(d...), ps)
        gs_vec = typeof(gs[ps[1]][1])[]
        for p in ps
            for xi in gs[p]
                push!(gs_vec, xi)
            end
        end
        opt.r = opt.decay*opt.r + (1-opt.decay)*sum(gs_vec.*gs_vec)
        cnt=0
        for p in ps
            for i in 1:prod(size(p))
                cnt=cnt+1
                p[i] = p[i] - opt.lr / ( eps(Float64)+sqrt(opt.r) ) * gs_vec[cnt]
            end
        end
    end
end

# adam
# adam == sgdm + adadelta
function train!(m, data, loss, opt::adam)
    ps = params(m)
    for d in data
        gs=gradient(()->loss(d...), ps)
        gs_vec = typeof(gs[ps[1]][1])[]
        for p in ps
            for xi in gs[p]
                push!(gs_vec, xi)
            end
        end
        if opt.m_pre==-1                   # 上次的动量，第一次调用时初始化为0
            opt.m_pre=zeros(size(gs_vec,1))
        end
        opt.r = opt.decay*opt.r + (1-opt.decay)*sum(gs_vec.*gs_vec)
        cnt=0
        for p in ps
            for i in 1:prod(size(p))
                cnt=cnt+1
                p[i] = p[i] - opt.lr / ( eps(Float64)+sqrt(opt.r) ) * ( opt.beta*opt.m_pre[cnt] + (1-opt.beta)*gs_vec[cnt] )
            end
        end
        opt.m_pre=deepcopy(gs_vec)
    end
end

# nadam
# nadam == sgda + adadelta
function train!(m, data, loss, opt::nadam)
    ps = params(m)
    for d in data
        if !(opt.m_pre==-1)
            cnt=0
            for p in ps
                for i in 1:prod(size(p))
                    cnt=cnt+1
                    p[i] = p[i] - opt.lr* opt.beta * opt.m_pre[cnt]
                end
            end
        end
        gs=gradient(()->loss(d...), ps)
        gs_vec = typeof(gs[ps[1]][1])[]
        for p in ps
            for xi in gs[p]
                push!(gs_vec, xi)
            end
        end
        if opt.m_pre==-1                    # 上次的动量，第一次调用时初始化为0
            opt.m_pre=zeros(size(gs_vec,1))
        end
        opt.r = opt.decay*opt.r + (1-opt.decay)*sum(gs_vec.*gs_vec)
        cnt=0
        for p in ps
            for i in 1:prod(size(p))
                cnt=cnt+1
                p[i] = p[i] - opt.lr / ( eps(Float64)+sqrt(opt.r) ) * ( opt.beta*opt.m_pre[cnt] + (1-opt.beta)*gs_vec[cnt] )
            end
        end
        opt.m_pre=deepcopy(gs_vec)
    end
end
