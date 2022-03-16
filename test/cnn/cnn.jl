using TrapDpLning
using MLDatasets
using Plots

x, y = MNIST.traindata(Float32)
if size(size(x), 1)==3
    x=reshape(x, (size(x,1),size(x,2),1,size(x,3)))
end
# one-hot
# 转换label，0~9转换为长度10的向量，0转换为[0,0,0,0,0,0,0,0,0,1]
if size(size(y), 1)==1
    tmp=[]
    for yi in y
        e=zeros(10)
        if yi==0
            e[10]=1
        else
            e[yi]=1
        end
        push!(tmp, e)
    end
    y=tmp
end
x_train = x[:,:,:,1:30000]
y_train = y[1:30000]
x_test = x[:,:,:,50001:52222]
y_test = y[50001:52222]

struct relu
end
function (m::relu)(x)
    x = x/2 + abs.(x)/2
end
struct softmax
end
function (m::softmax)(x)
    s=sum(exp.(x))
    [exp(xi)/s for xi in x]
end

m=chain(cnn(5,5,1,5), cnn_d(5,5,5,20), maxpool((5,5)), fullconnect(4*4*20,10), softmax())
data=zip([x_train[:,:,:,i:i] for i in 1:size(x_train,4)], y_train)
loss(x,y)=crossentropy(m(x), y)
opt=adam(0.1)

# 曲线
res_show=[[],[],[]]

# 计算accu
function cal_accu(start, stop)
    cnt = 0
    x_accu = x[:,:,:,start:stop]
    y_accu = y[start:stop]
    for i in 1:(stop-start+1)
        if sum( abs.( m(x_accu[:,:,:,i:i]) .- y_accu[i] ) ) < 1 # 跟label相同的类别概率超过0.5
            cnt+=1
        end
    end
    cnt/(stop-start+1)
end

# 训练1:30000张图片，测试50001:52222张图片
batch=100
TRAIN_TIMES=300
for k in 1:300

    # 学习率衰减, 0.05 -> 0.001
    opt = adam(0.05 - (0.05-0.001)*k/TRAIN_TIMES)

    x_train=x[:,:,:,1+(k-1)*batch:k*batch]
    y_train=y[1+(k-1)*batch:k*batch]
    data=zip([x_train[:,:,:,i:i] for i in 1:size(x_train,4)], y_train)

    train!(m,data,loss,opt)

    show_loss = sum( [loss(x_test[:,:,:,i:i], y_test[i]) for i in 1:size(y_test,1)] )
    show_accu = cal_accu(50001, 52222)

    push!(res_show[1], show_loss)
    push!(res_show[2], show_accu)

    println(k, ": loss = ", show_loss, " accu = ", show_accu)
end

# 曲线
default(show=true)
plotly()
plot(res_show[1:1], linewidth=3, xlabel="TRAIN_TIMES", ylabel="loss(test)", title="test_cnn_loss")
plot(res_show[2:2], linewidth=3, xlabel="TRAIN_TIMES", ylabel="accu(test)", title="test_cnn_accu")
