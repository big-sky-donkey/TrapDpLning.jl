# 将txt文件，转换为，三维Vector，使用【@#￥】区分维度

function file2vvv(filename)
    # 读入字符串
    io = open(filename)
    s=read(io, String)

    # 运输工具，目的是制造三维Vector
    res=Vector{Vector{Float64}}[]
    v2=Vector{Float64}[]
    v1=Float64[]

    i=1
    cnt=0
    len=length(s)

    while i<=len
        if s[i]=='@'
            push!(v1, Float64( str2num( s[i-cnt:i-1] ) ) )
            cnt=0
        elseif s[i]=='#'
            push!(v2, v1)
            v1=Float64[]
        elseif s[i]=='$'
            push!(res, v2)
            v2=Vector{Float64}[]
        else
            cnt+=1
        end
        i+=1
    end

    res
end