# 收集模型的参数，写模型时需要多派trainable，表明需要收集的参数

function params(m)
    ps = Params()
    ps_collect!(ps, m)
    return ps
end

# 对于AbstractArray，收集参数
# 对于其他（比如网络层），递归收集参数
function ps_collect!(ps, x, idd=IdDict()) 
    # idd记录已收集的参数（以及layer），防重复
    # 语法：函数递归调用，idd被传递下去，IdDict对地址objectid防重复
    if x in keys(idd)
        return
    end
    idd[x]=nothing
    for xi in trainable(x)
        ps_collect!(ps, xi, idd)
    end
end

function ps_collect!(ps, x::AbstractArray, idd=IdDict())
    # Zygote重写了push!，收集参数
    push!(ps, x)
end

# 当没有某类型的多重派发trainable时，调用这个函数，不收集任何参数
function trainable(m)
    ()
end
