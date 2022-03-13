# 池化层：保留主要特征的同时，减小计算量、防止过拟合
# more: 需要舍弃的尺寸>=2时，目前是舍弃右下方，如果上下左右分担舍弃，会更合理

struct maxpool
    window  # window的每个尺寸不能比x大，但是不必整除，不能整除的时候舍弃x中右方与下方的多余的
    padding
    stride  # 移动时，步长，取window和stride中某尺寸上较大值
    function maxpool(window=(1,1), padding=(0,0,0,0), stride=(1,1))
        new(window, padding, stride)
    end
end

# x的4个sz: x行数，x列数，x的chn，x的batch
# window: 1上下 2左右
# padding: 1上 2下 3左 4右
# stride: 1上下 2左右

function m_max(x)
    res=x[1]
    for xi in x
        if res<xi
            res=xi
        end
    end
    res
end

function m_max(a,b)
    (a>b) ? a : b
end

function (m::maxpool)(x)
    window, padding, stride = m.window, m.padding, m.stride

    x=[xi>padding[1]&&xi<=size(x,1)+padding[1]&&xj>padding[3]&&xj<=size(x,2)+padding[3] ? x[xi-padding[1],xj-padding[3],xc,index] : 0.0 
        for xi in 1:size(x,1)+padding[1]+padding[2],
            xj in 1:size(x,2)+padding[3]+padding[4],
            xc in 1:size(x,3),
            index in 1:size(x,4)]

    [m_max(x[yi:yi+window[1]-1, yj:yj+window[2]-1, yc, index])
        for yi in 1:m_max(window[1],stride[1]):size(x,1)-window[1]+1, 
            yj in 1:m_max(window[2],stride[2]):size(x,2)-window[2]+1, 
            yc in 1:size(x,3),
            index in 1:size(x,4)]
end

function trainable(m::maxpool)
    ()
end




# # 【备份】
# # pool:
# function mp(x)
#     x=[xi>padding[1]&&xi<=size(x,1)+padding[1]&&xj>padding[3]&&xj<=size(x,2)+padding[3] ? x[xi-padding[1],xj-padding[3],xc,index] : 0.0 
#         for xi in 1:size(x,1)+padding[1]+padding[2],
#             xj in 1:size(x,2)+padding[3]+padding[4],
#             xc in 1:size(x,3),
#             index in 1:size(x,4)]

#     [m_max(x[yi:yi+window[1]-1, yj:yj+window[2]-1, yc, index])
#         for yi in 1:m_max(window[1],stride[1]):size(x,1)-window[1]+1, 
#             yj in 1:m_max(window[2],stride[2]):size(x,2)-window[2]+1, 
#             yc in 1:size(x,3),
#             index in 1:size(x,4)]
# end
