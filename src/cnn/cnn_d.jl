# 深度可分离卷积
# 用cin个kernel得到cin个图片，用cin*cout矩阵进行线性组合，得到res，参数量=(x*y*cin+cin*cout)

struct cnn_d
    w        # windows（窗口们），3维数组（x长度, y长度, in）
    lc       # linear combination
    padding
    dilation
    stride
    function cnn_d(w1, w2, in, out; padding=(0,0,0,0), dilation=(1,1), stride=(1,1))
        w = (rand(Float64, w1, w2, in).-0.5)*0.08 # -0.04~0.04
        lc = rand(Float64, out, in)
        # 目的：lc每行的和设置为1
        # 实现：临时数组s，记录lc每行和，然后将lc除以s的对应元素
        s = [ sum( [lc[i,j] for j in 1:size(lc,2)] ) for i in 1:size(lc,1)]
        lc = [ lc[i,j]/s[i] for i in 1:size(lc,1), j in 1:size(lc,2)]
        new(w, lc, padding, dilation, stride)
    end
end

function (m::cnn_d)(x)
    w, lc, padding, dilation, stride = m.w, m.lc, m.padding, m.dilation, m.stride

    # 自适应padding，从而保证conv后尺寸不变
    if padding=="same"
        padding=(Int(floor((stride[1] * size(x,1) - size(x,1) + (size(w,1)+(size(w,1)-1)*(dilation[1]-1)) -1+1-1)/2)),
                Int(floor((stride[1] * size(x,1) - size(x,1) + (size(w,1)+(size(w,1)-1)*(dilation[1]-1)) -1+1-1)/2))+(size(w,1)%2==0),
                Int(floor((stride[2] * size(x,2) - size(x,2) + (size(w,2)+(size(w,2)-1)*(dilation[2]-1)) -1+1-1)/2)),
                Int(floor((stride[2] * size(x,2) - size(x,2) + (size(w,2)+(size(w,2)-1)*(dilation[2]-1)) -1+1-1)/2))+(size(w,2)%2==0)
            )
    end
    # 如何实现"same"？需要求出所需增加的空行，思路是列方程来解
    #
    # （1）设p12是竖直方向所需添加空行，可以算出卷积核左上角的可活动范围，这个范围的尺寸就是size(x,1)，列出方程：
    #      【1 : stride[1] : 【size(x,1)+p12 - (size(w,1)+(size(w,1)-1)*(dilation[1]-1)) + 1】】范围的元素数量length == size(x,1)，
    #       # p12表示padding[1]+padding[2]，(size(w,1)+(size(w,1)-1)*(dilation[1]-1))表示膨胀后的窗口的竖直长度
    # （2）start+step*length，得到最后一个step的尾后（也就是stop取它时刚好使得卷积后size大了1），想要length准确，只要stop=这个尾后-1来解方程
    #      （因为这里length刚好减少了1，而stop减少1时，length只可能减少1或不减少）
    #      size(x,1)+p12 - (size(w,1)+(size(w,1)-1)*(dilation[1]-1)) + 1 == start+stride[1]*size(x,1)-1
    # （3）算出p12后，除以2得到p1和p2，如果w的尺寸是偶数，则p12是奇数，则在p1或p2上+1
    # （4）p34同理可求
    # 总结：目的是求p12使得这个range的length达到目标，显然随着p12增大1，length是增大的（且只能增大1或不变）
    #       所以求出刚好让length超标的p12值，再上面-1，得到理想的p12

    # padding
    x=[xi>padding[1]&&xi<=size(x,1)+padding[1]&&xj>padding[3]&&xj<=size(x,2)+padding[3] ? x[xi-padding[1],xj-padding[3],xc,index] : 0.0 
        for xi in 1:size(x,1)+padding[1]+padding[2],
            xj in 1:size(x,2)+padding[3]+padding[4],
            xc in 1:size(x,3),
            index in 1:size(x,4)
    ]
    
    # dilation
    w=[(wi-1)%dilation[1]==0&&(wj-1)%dilation[2]==0 ? w[(Int)((wi-1)/dilation[1])+1,(Int)((wj-1)/dilation[2])+1,cin] : 0.0 
        for wi in 1:size(w,1)+(size(w,1)-1)*(dilation[1]-1),
            wj in 1:size(w,2)+(size(w,2)-1)*(dilation[2]-1),
            cin in 1:size(w,3)
    ]

    # conv
    # sum：进行卷积运算的求和(第12维)，不进行频道全连接的求和(第3维)，而是对第3维获得cin个卷积结果，用于接下来的线性组合
    to_be_lc=[sum(x[yi:yi+size(w,1)-1, yj:yj+size(w,2)-1, cin, index] .* w[1:size(w,1), 1:size(w,2), cin])
        for yi in 1:stride[1]:size(x,1)-size(w,1)+1, 
            yj in 1:stride[2]:size(x,2)-size(w,2)+1, 
            cin in 1:size(w,3), 
            index in 1:size(x,4)
    ]
    # 线性组合：cin->cout
    # sum：对cin求和
    [ sum( to_be_lc[i,j,1:size(lc,2),index] .* lc[cout, 1:size(lc,2)] )
        for i in 1:size(to_be_lc,1),
            j in 1:size(to_be_lc,2),
            cout in 1:size(lc,1),
            index in 1:size(x,4)
    ]
end

function trainable(m::cnn_d)
    (m.w, m.lc)
end

