using TrapDpLning
using DelimitedFiles

data = readdlm("F:\\TrapDpLning\\test\\cptnn\\data.txt") # 需要修改地址
data = normalization(data)
m = som(8,8,size(data,1),0.01,1.0,"Hexagon")

for i in 1:10
    train!(m,data)
end

res=zeros(8,8)
for i in 1:size(data,2)
    co = m(data[:,i])
    res[co[1],co[2]] += 1
end

# 求出最多数的类（data中有最多的数据距离它最近）
biggest=-1
biggest_coodi=[-1,-1]
for i in size(res,1)
    for j in size(res,2)
        if res[i,j] > biggest
            biggest = res[i,j]
            biggest_coodi=[i,j]
        end
    end
end
# 这个是最多数的类
print(m.w[biggest_coodi[1],biggest_coodi[2],:])

# more：可以分类
