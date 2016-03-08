module Synthetic

abstract SyntheticTask
abstract RegressionTask <: SyntheticTask
abstract ClassificationTask <: SyntheticTask

Base.rand{T<:SyntheticTask}(task::T, n::Int) = eltype(task)[rand(task) for i = 1:n]

include("copy.jl")
include("add.jl")
include("mog.jl")
include("xor.jl")

end # module Synthetic