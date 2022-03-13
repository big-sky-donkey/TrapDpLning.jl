# 径向基神经网络
# 局部映射：w中的一个值，只能影响输出m(x)中的一个值
# 理论上，rbnn是连续函数的最佳逼近
struct rbnn
    w      # in*out
    center # in, 中心点可以随机初始化，也可以根据输入初始化
    rbf    # radial basis function
    function rbnn(in, out, rbf="gauss", delta=1.0)
        w = rand(in, out) .- 0.5
        center = rand(in) .- 0.5
        str2func = Dict(
                "gauss" => ( x -> exp.( (-1.0) * (1/(2*delta^2)) * x.^2 ) ),
                "reflected_sigmoidal" => ( x -> 1 / (1 + exp.( (1/(delta^2)) * x.^2 )) ),
                "inverse_multiquadrics" => ( x -> 1 / sqrt.( x.^2 .+ delta^2 ) )
            )
        new(w, center, str2func[rbf])
    end
end

function (m::rbnn)(x)
    ( ones(1,size(m.center,1)) * ( m.w .* m.rbf.(x - m.center) ) )'
end

function trainable(m::rbnn)
    (m.w, m.center)
end
