using TrapDpLning
using DelimitedFiles

struct softmax
end
function (m::softmax)(x)
    s=sum(exp.(x))
    [exp(xi)/s for xi in x]
end

m=chain(birnn(lstm(12,80),gru(12,80)), perceptron(80,9), softmax())

# x三维，y二维
x=file2vvv("F:\\TrapDpLning\\test\\rnn\\x.txt") # 需要修改地址
y=readdlm("F:\\TrapDpLning\\test\\rnn\\y.txt")  # 需要修改地址
y=y'
y=[y[:,j] for j in 1:size(y,2)]

# x由3D-vec转为vec{mat}，按列存储，转换后方便按列遍历
x = [ [ x[k][i][j] for i in 1:size(x[k],1), j in 1:size(x[k][1],1) ] for k in 1:size(x,1)]

# one-hot
# 转换label，1~9转换为长度10的向量，比如[1,0,0,0,0,0,0,0,0]
if size(y[1],1)==1
    tmp=[]
    for yi in y
        e=zeros(9)
        e[Int(yi[1])]=1
        push!(tmp, e)
    end
    y=tmp
end

# shuffle，等差遍历，凑合一下，凑合能用
# 这样能shuffle吗？
# 方法：31与270互质，每次前进31位然后对270求余，从而打乱
# 严谨：每次增31，想要有重复的话，两个重复选取之间的差必须被270和31整除，刚好270个新的对应270个旧的
tmpx=deepcopy(x)
tmpy=deepcopy(y)
for i in 1:270
    tmpx[i]=x[((i-1)*31)%270+1] # 互质，改变顺序而不重复
    tmpy[i]=y[((i-1)*31)%270+1]
end
x=tmpx
y=tmpy

# split
x_train=x[1:220]
y_train=y[1:220]
x_test=x[221:270]
y_test=y[221:270]

data=zip(x_train, y_train)

function loss(x,y)
    # 状态置零
    m.layers[1].rnn1.state = zeros(size(m.layers[1].rnn1.state,1))
    m.layers[1].rnn2.state = zeros(size(m.layers[1].rnn2.state,1))
    crossentropy( [ m(x[:,j], x[:,size(x,2)+1-j]) for j in 1:size(x,2) ][end], y )
end
opt = adam(0.01)

for k in 1:100
    train!(m, data, loss, opt)
    show_loss = sum( [ loss(x_test[i], y_test[i]) for i in 1:size(y_test,1) ] )
    println(k, ": loss = ", show_loss)
end

# 查看测试数据的分类正误
for a in 1:size(x_test,1)
    m.layers[1].rnn1.state = zeros(size(m.layers[1].rnn1.state,1))
    m.layers[1].rnn2.state = zeros(size(m.layers[1].rnn2.state,1))
    println(sum(abs.([ m(x_test[a][:,j], x_test[a][:,size(x_test[a],2)+1-j]) for j in 1:size(x_test[a],2) ][end]-y_test[a]) )<1)
end