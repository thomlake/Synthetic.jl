# Sequential XOR
type XORTask <: ClassificationTask
    range::UnitRange{Int}
end

XORTask(t::Int) = XORTask(t:t)

Base.size(task::XORTask) = (1, 1)

function Base.size(task::XORTask, i::Int)
    if i == 1
        return 1
    elseif i == 2
        return 1
    else
        error("$i is not a valid index. Use 1 (output dim) or 2 (input dim)")
    end
end

Base.eltype(task::XORTask) = Tuple{Vector{Int},Vector{Int}}

function Base.rand(task::XORTask)
    bits = rand(0:1, rand(task.range))
    x = bits + 1
    y = (cumsum(bits) % 2) + 1
    return x, y
end
