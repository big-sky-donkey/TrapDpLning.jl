module TrapDpLning


using Zygote         # 求导
using MLDatasets     # 下载数据集
using DelimitedFiles # 读取txt
using Plots          # 画图


# 转换工具，提取需要训练的参数，归一化，网络串联
export str2num, file2vvv
export params
export normalization
export chain
include("util/str2num.jl")
include("util/file2vvv.jl")
include("util/params.jl")
include("util/normalization.jl")
include("util/chain.jl")

# 一步神经网络
# 感知机，径向基网络
export perceptron, rbnn
include("nn/perceptron.jl")
include("nn/rbnn.jl")

# 卷积神经网络
# 卷积，深度可分离卷积，最大池化，全连接，softmax定义在测例中
export cnn, cnn_d, maxpool, fullconnect
include("cnn/cnn.jl")
include("cnn/cnn_d.jl")
include("cnn/pool.jl")
include("cnn/fullconnect.jl")

# 循环神经网络
# rnn, lstm, gru, 双向可双拼rnn
export rnn, lstm, gru, birnn
include("rnn/util/funcs.jl")
include("rnn/rnn.jl")
include("rnn/lstm.jl")
include("rnn/gru.jl")
include("rnn/birnn.jl")

# 自组织映射神经网络
export som
export train!
include("som/util/distance_func.jl")
include("som/util/neighbor_func.jl")
include("som/som.jl")
include("som/train.jl")

# 损失函数
export mse, crossentropy
include("loss/loss.jl")

# 优化算法
export sgd, sgdm, sgda, adagrad, adadelta, adam, nadam
include("optimise/optimise.jl")

# 训练
export train!
include("train/train.jl")


end # module
