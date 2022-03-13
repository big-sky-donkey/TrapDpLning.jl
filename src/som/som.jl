# 自组织映射som
# 应用：聚类、可视化
# 计算过程：
# （1）归一化：data的列向量
# （2）竞争：选出wj中与x相似性最大的wj（相似性调用各种距离函数来计算），作为winner
# （3）输出、更新：输出winner、更新w（取决于到winner的距离）
# 更新公式：w[i][j]列向量 += lr * 邻居打折函数 * （data的列向量 - w[i][j]列向量）

struct som        # som, sofm, Kohonen
    w             # x*y*in, xy表示参与竞争的一个向量的index
    coodi         # 3维vec, x*y*2, 表示二维空间的每个点的坐标, 根据拓扑结构初始化, 后不再改变
    lr
    sigma         # 扩展系数，用于neighbor_func
    topology      # 拓扑，四边形或者六边形
    neighbor_func # 输入coodi,winner,sigma，输出一个矩阵表示更新打几折（直接根据物理距离计算，而非调距离函数），用于更新权重
    distance_func # 输入两个vec，输出它们之间的距离，用于确定winner

    function som(x, y, in, lr=0.1, sigma=1.0, topology="Rectang", neighbor_func="Gauss", distance_func="Euclidean")
        w=rand(x,y,in)*2 .-1
        # 调整初始化权重：对第三个维度的向量，分别除以它的2-范数
        for i in 1:size(w,1)
            for j in 1:size(w,2)
                w[i,j,:] /= sqrt(sum(w[i,j,:].^2))
            end
        end

        # 生成拓扑
        if topology=="Rectang"
            coodi=zeros(x,y,2)
            for i in 1:x
                for j in 1:y
                    coodi[i,j,:]=[i,j]
                end
            end
        elseif  topology=="Hexagon"
            coodi=zeros(x,y,2)
            for i in 1:x
                for j in 1:y
                    coodi[i,j,:]=[i-(j%2==0)/2,j/sqrt(3.0)] # 保证非边缘的点到附近6个点的距离相等，但是y方向被压缩了根号3，影响不大
                end
            end
        end

        # 函数名 -> 函数对象
        str2neighbor=Dict(
            "Gauss"=>Gauss_neighbor
        )
        str2distance=Dict(
            "Manhattan"=>Manhattan_distance,
            "Euclidean"=>Euclidean_distance,
            "Chebyshev"=>Chebyshev_distance,
            "Cosine"=>Cosine_distance
        )

        new(w, coodi, lr, sigma, topology, str2neighbor[neighbor_func], str2distance[distance_func])
    end
end

function (m::som)(x)
    # 求出距离一个数据x最近的wij，返回[i,j]
    w,distance_func = m.w, m.distance_func
    distance=-1
    winner=[-1,-1] # 显然可以保证这俩会被更新至少一次
    for i in 1:size(w,1)
        for j in 1:size(w,2)
            if distance_func(w[i,j,:], x) > distance
                distance = distance_func(w[i,j,:], x)
                winner=[i,j]
            end
        end
    end
    winner
end
