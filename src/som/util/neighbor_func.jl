# 用于som
function Gauss_neighbor(coodi, winner, sigma)
    # 计算一个打折矩阵（用于竞争学习中的更新参数）
    # 它跟拓扑矩阵中各个元素到winner的物理距离有关，越近则res越大
    # 距离矩阵，[abs.(coodi[i,j,:]-winner) for i in 1:size(coodi,1), j in 1:size(coodi,2)]
    # 注意：对坐标prod然后exp，不等价于，exp然后对坐标prod
    [prod(exp.( (-1) * ( ( [abs.(coodi[i,j,k]-winner[k]) for i in 1:size(coodi,1), j in 1:size(coodi,2), k in 1:2] ).^2 ) / (2*sigma*sigma) )[p,q,:])
    for p in 1:size(coodi,1), q in 1:size(coodi,2)]
end
