# TrapDpLning.jl
初入机器学习，造轮子上瘾，浪费时间精力却沉迷其中，陷入了造轮子的陷阱，没有意义，还不如多看看新论文或者造更棒的模型。

陷阱深度学习TrapDpLning.jl



nn：

（1）感知机，径向基神经网络（擅长逼近函数），som（聚类）。

cnn：（图像分类）

（1）conv，实现了可调整的window、padding、dilation、stride，设置了可选的same padding自动补齐conv前后的图片尺寸。

（2）maxpool，实现了可调整的window、padding、stride。

（3）fullconnect，将4维数组数据转换为1维。

（4）因为自动求导必须使用不可变的数组，所以语法上使用了【for解析式、?:分支】，可读性差。

（5）运行速度太慢，依次测试，发现模型计算不慢，慢在参数太多，可以使用“深度可分离卷积”减少参数量，可以加速并减轻过拟合，还需要修改模型搭建，让全连接层的参数少一些。

rnn：（序列分类）

（1）实现了rnn、lstm、gru，对象中保有一个state，遇到新的序列时需要置零，rnn函子的参数是序列中的一个元素。

（2）实现了双向的birnn，可以任选两个rnn系列的网络一正一反遍历序列。

（3）其他方法：手写函数导入matlab中的三维数组，等差遍历shuffle凑合一下，softmax与one-hot。

loss：

（1）mse（回归）、crossentropy（分类）。

opt：

（1）sgd系列（sgd、Momentun、Nesterov），adapt系列（adagrad、rmsprop），合体系列（adam、nadam）。

util：

（1）chain：把多个神经网络串起来。

（2）提取参数：将模型中需要训练的参数提取到Zygote.jl的对象Params，需要每个模型增加函数trainable的多态，表示出需要求导的参数。如果没有写trainable函数，则不提取参数。

（3）归一化。

（4）转换：字符串->数字（julia竟然没有API），txt->三维数组（API只支持一维二维，rnn系列需要三维数组的数据）。



【EXAMPLES】【EXAMPLES】:

（1）birnn（正向lstm、反向gru，隐藏层80个节点）

问题：来自matlab的数据集japaneseVowelsTrainData（带label）分类。

设置：loss使用交叉熵loss，opt使用adam（学习率0.006->0.002，随便设的）

效果：训练集准确率97%，测试集准确率92%



![066fc0088bdff8fdbd98ac5eaa4c632](https://user-images.githubusercontent.com/81020046/158574975-fada682a-e8ef-472b-9114-b9d4689d1b5b.png)



![6a8f2d9a7603f08e8dc157543f48a73](https://user-images.githubusercontent.com/81020046/158574999-684e40f0-b28f-4d52-9c9d-2d248941cb6f.png)



![83dac0983fdbc10bc7c3fb44ba81b75](https://user-images.githubusercontent.com/81020046/158575006-2539757a-b739-4589-b39a-301c36026ae6.png)

（2）cnn（ cnn（卷积核5*5）








