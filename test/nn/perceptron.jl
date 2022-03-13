using TrapDpLning
using Plots

m=chain(perceptron(3, 5), perceptron(5, 7))
data=[(ones(3), ones(7).*5)]
loss(x,y) = mse(m(x),y)
opt = sgd()

TRAIN_TIMES=80
x_test=ones(3)

res_show=zeros(TRAIN_TIMES+1, size(data[1][2])[1])'
for i in 1:size(data[1][2])[1]
    res_show[i]=m(x_test)[i]
end
for j in 1:TRAIN_TIMES
    train!(m, data, loss, opt)
    for i in 1:size(data[1][2])[1]
        res_show[j*size(data[1][2])[1]+i]=m(x_test)[i]
    end
end

m(x_test)

default(show=true)
plotly()
plot(res_show', linewidth=3, title="test_perceptron")
