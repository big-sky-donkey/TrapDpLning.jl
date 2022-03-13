using TrapDpLning
using MLDatasets

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
x_train = x[:,:,:,1:50]
y_train = y[1:50]
x_test = x[:,:,:,2001:2500]
y_test = y[2001:2500]

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

m=chain(cnn(5,5,1,3), cnn_d(5,5,3,10), maxpool((5,5)), fullconnect(4*4*10,10), softmax())
data=zip([x_train[:,:,:,i:i] for i in 1:size(x_train,4)], y_train)
loss(x,y)=crossentropy(m(x), y)
opt=adam(0.1)

for k in 1:100
    train!(m,data,loss,opt)
    show_loss_1 = sum( [loss(x_train[:,:,:,i:i], y_train[i]) for i in 1:size(y_train,1)] )
    show_loss_2 = sum( [loss(x_test[:,:,:,i:i], y_test[i]) for i in 1:size(y_test,1)] )
    println(k, ": loss_train = ", show_loss_1, "  loss_test = ", show_loss_2)
end

# 计算accu
start, stop, cnt = 50001, 51111, 0
x_accu = x[:,:,:,start:stop]
y_accy = y[start:stop]
for i in 1:(stop-start+1)
    if sum( abs.( m(x_accu[:,:,:,i:i]) .- y_accy[i] ) ) < 1.2 # 跟label相同的类别概率超过0.4
        cnt+=1
    end
end
print("accu = ", cnt/(stop-start+1))
