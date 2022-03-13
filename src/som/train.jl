# som
function train!(m::som, data)
    w, coodi, lr, sigma, distance_func = m.w, m.coodi, m.lr, m.sigma, m.distance_func

    for k in 1:size(data,2)
        # d = data[:,k] # 拷贝到元素了，多余，可以把data[:,k]写到下面直接访问
        distance=-1
        winner=[-1,-1]  # 显然可以保证这俩会被更新至少一次
        for i in 1:size(w,1)
            for j in 1:size(w,2)
                if distance_func(w[i,j,:], data[:,k]) > distance
                    distance = distance_func(w[i,j,:], data[:,k])
                    winner=[i,j]
                end
            end
        end

        for i in 1:size(w,1)
            for j in 1:size(w,2)
                # 选择data-w，而非data，来更新w，直观上看是很合理的
                w[i,j,:] = w[i,j,:] + lr * Gauss_neighbor(coodi, winner, sigma)[i,j] * (-w[i,j,:] .+ data[:,k])
            end
        end
        # 思考语法：w的权重值为什么能变？
        # 传进来m的拷贝，指向真实的m.w，w是m.w的拷贝，指向真实的m.w的元素，w[]访问到了真实的元素
    end
end
