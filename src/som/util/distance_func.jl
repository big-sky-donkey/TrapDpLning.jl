# 用于cptnn, som
# 计算两个vec之间的距离

# sum(|xi-yi|^p)^(1/p)
# 与数据分布无关，res不携带数据分布的信息
# 需要归一化（否则不同方向幅值不同，对res的作用大小不同）

# p=1(Manhattan), p=2(Euclidean), p=∞(Chebyshev)
function Manhattan_distance(x, y)
    sum([abs(x[i]-y[i]) for i in 1:size(x,1)])
end
function Euclidean_distance(x, y)
    sqrt(sum([(x[i]-y[i])^2 for i in 1:size(x,1)]))
end
function Chebyshev_distance(x, y)
    res=0 # 最大值
    [(abs(x[i]-y[i])>res) ? abs(x[i]-y[i]) : res for i in 1:size(x,1)][end]
end


# 余弦相似度
function Cosine_distance(x, y)
    x_norm2=sqrt( sum( [x[i]*x[i] for i in 1:size(x,1)] ) )
    y_norm2=sqrt( sum( [y[i]*y[i] for i in 1:size(y,1)] ) )
    sum([x[i]*y[i] for i in 1:size(x,1)]) / ( x_norm2 * y_norm2 + eps(Float64))
end
