# mse, 可以用于回归问题
function mse(x, y)
    sum( (x .- y).^2 ) / length(x)
end

# 交叉熵
# 对于分类问题，mseloss对ps的函数，有许多局部极值点，非凸优化问题，难以使用梯度下降，使用交叉熵loss则是凸优化
# label是[0,0,0,1,0]的形式，对应参数y
function crossentropy(x, y)
    sum( (-1)*x.*log.(y .+ eps(Float64)) ) / length(x) # 必须加eps
end
