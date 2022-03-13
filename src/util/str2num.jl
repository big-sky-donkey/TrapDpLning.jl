# 将"-123.123"转换为-123.123
# 如果是"."，那么会得到0，不一定是Float

function str2num(s)
    idx=0
    for i in 1:length(s)
        if s[i]=='.'
            idx=i
            break
        end
    end
    s1=s[1:idx-1]
    s2=s[idx+1:length(s)]
    res1=0
    res2=0
    first=1 # 判断正负
    # 不必讨论s1,s2为空的case
    for c in s1
        if c=='-'
            first=-1
            continue
        end
        res1*=10
        res1+=c-'0'
    end
    for c in reverse(s2)
        res2+=c-'0'
        res2/=10
    end
    (res1 + res2) * first
end