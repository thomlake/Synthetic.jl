"""
Synthetic Copy Task
title: Orthogonal RNNs and Long-Memory Tasks
paper: http://arxiv.org/pdf/1602.06662v1.pdf
"""
immutable CopyTask <: ClassificationTask
    delay::UnitRange{Int}
    capacity::UnitRange{Int}
    dim::Int
    function CopyTask(delay, capacity, dim)
        dim > 2 || error("dim must be greater than 2")
        (last(capacity) >= first(capacity) && last(capacity) > 0) || error("invalid capacity range $capacity")
        (last(delay) >= first(delay) && last(delay) > 0) || error("invalid delay range $delay")
        return new(delay, capacity, dim)
    end
end

CopyTask(delay::Int=100, capacity::Int=10, dim::Int=10) = CopyTask(delay:delay, capacity:capacity, dim)

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
    t1 = rand(task.capacity)
    t2 = rand(task.delay)
    
    prefix = rand(1:task.dim - 2, t1)   # sequence to remember
    blanks = fill(task.dim - 1, t2 - 1) # blanks
    flag = task.dim                     # flag indicating recall should begin
    suffix = fill(task.dim - 1, t1)     # blanks
    
    x = vcat(prefix, blanks, flag, suffix)
    y = vcat(fill(task.dim - 1, t1 + t2), prefix)
    
    return x, y
end
