"""
Synthetic Addition Task
title: Orthogonal RNNs and Long-Memory Tasks
paper: http://arxiv.org/pdf/1602.06662v1.pdf
"""
immutable AddTask <: RegressionTask
    range::UnitRange{Int}
end

AddTask(t::Int) = AddTask(t:t)

Base.size(task::AddTask) = (2, 1)

function Base.size(task::AddTask, i::Int)
    if i == 1
        return 2
    elseif i == 2
        return 1
    else
        error("$i is not a valid index. Use 1 (output dim) or 2 (input dim)")
    end
end

Base.eltype(task::AddTask) = Tuple{Vector{Vector{Float64}},Float64}

function Base.rand(task::AddTask)
    steps = rand(task.range)
    i1 = rand(1:steps)
    i2 = i1
    while i1 == i2
        i2 = rand(1:steps)
    end
    n1 = rand()
    n2 = rand()
    target = n1 + n2
    inputs = Vector{Float64}[]

    for i = 1:steps
        x = if i == i1
            [n1, 1]
        elseif i == i2
            [n2, 1]
        else
            [rand(), 0]
        end
        push!(inputs, x)
    end

    return inputs, target
end
