"""
Synthetic Copy Task
title: Orthogonal RNNs and Long-Memory Tasks
paper: http://arxiv.org/pdf/1602.06662v1.pdf
"""
immutable CopyTask <: ClassificationTask
    dim::Int
    prefix_range::UnitRange{Int}
    suffix_range::UnitRange{Int}
    function CopyTask(d, pr, sr)
        d > 2 || error("d must be greater than 2")
        (last(pr) >= first(pr) && last(pr) > 0) || error("not enough prefix timesteps")
        (last(sr) >= first(sr) && last(sr) > 0) || error("not enough suffix timesteps")
        return new(d, pr, sr)
    end
end

CopyTask(t::Int) = CopyTask(10, 10:10, t:t)


Base.size(task::CopyTask) = (task.dim, task.dim)

function Base.size(task::CopyTask, i::Int)
    if i == 1
        return task.dim
    elseif i == 2
        return task.dim
    else
        error("$i is not a valid index. Use 1 (output dim) or 2 (input dim)")
    end
end

Base.eltype(task::CopyTask) = Tuple{Vector{Int},Vector{Int}}

function Base.rand(task::CopyTask)
    t1 = rand(task.prefix_range)
    t2 = rand(task.suffix_range)
    
    prefix = rand(1:task.dim - 2, t1)   # sequence to remember
    blanks = fill(task.dim - 1, t2 - 1) # blanks
    flag = task.dim                     # flag indicating recall should begin
    suffix = fill(task.dim - 1, t1)     # blanks
    
    x = vcat(prefix, blanks, flag, suffix)
    y = vcat(fill(task.dim - 1, t1 + t2), prefix)
    
    return x, y
end
