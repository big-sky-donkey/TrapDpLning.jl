# 归一化
function normalization(data::Vector)
    # data是向量
    e=sum(data)/size(data,1)
    std=sqrt( sum( (data.-e).^2 ) / ( eps(Float64)+size(data,1)-1 ) )
    (data.-e) / (eps(Float64)+std)
end

function normalization(data::Matrix)
    # data是矩阵，列是特征，行是数据
    # e,std是每列的数据的均值和标准差构成的向量
    e=[sum(data[:,j])/size(data,1) for j in 1:size(data,2)]
    std=[sqrt( sum( (data[:,j].-e[j]).^2 ) / ( eps(Float64)+size(data,1)-1 ) ) for j in 1:size(data,2)]
    [(data[i,j]-e[j])/(eps(Float64)+std[j]) for i in 1:size(data,1), j in 1:size(data,2)]
end