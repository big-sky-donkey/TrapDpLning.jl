# 感知机
struct perceptron
    w
    b
    act
    function perceptron(in, out, act=identity)
        w = rand(out,in).-0.5
        b = zeros(out)
        new(w, b, act)
    end
end

function (m::perceptron)(x)
    m.act.( m.w * x + m.b )
end

function trainable(m::perceptron)
    (m.w, m.b)
end
