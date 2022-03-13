using TrapDpLning
using Plots

m=rbnn(11, 23, "inverse_multiquadrics") # "gauss"  "reflected_sigmoidal"  "inverse_multiquadrics"
loss(x, y) = mse(m(x), y)
opt = sgd()
data=[(ones(11), ones(23).*5)]

TRAIN_TIMES=80
x_test=ones(11)

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
plot(res_show', linewidth=3, title="test_rbnn")
