# Mixture of Gaussians
import Distributions: MixtureModel, Categorical, MultivariateNormal
import ConjugatePriors: NormalInverseWishart

immutable MixtureTask <: ClassificationTask
    model::MixtureModel
end

function MixtureTask(n_features::Int, n_classes::Int)
    d = NormalInverseWishart(zeros(n_features), 1, eye(n_features), 2 * n_features)
    model = MixtureModel([MultivariateNormal(rand(d)...) for i = 1:n_classes], Categorical(n_classes))
    return MixtureTask(model)
end

Base.size(task::MixtureTask) = (length(task.model.components[1]), length(task.model.prior))

function Base.size(task::MixtureTask, i::Int)
    if i == 1
        return length(task.model.components[1])
    elseif i == 2
        return length(task.model.prior)
    else
        error("$i is not a valid index. Use 1 (output dim) or 2 (input dim)")
    end
end

Base.eltype(task::MixtureTask) = Tuple{Vector{Float64},Int}

function Base.rand(task::MixtureTask)
    y = rand(task.model.prior)
    x = rand(task.model.components[y])
    return x, y
end
