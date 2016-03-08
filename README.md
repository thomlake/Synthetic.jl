# Synthetic.jl
Synthetic.jl is a Julia package for generating synthetic data. It is designed to be used for testing and sanity checking Machine Learning algorithms.

## Overview
Synthetic.jl defines an abstract type `SyntheticTask` and two abstract  subtypes: `RegressionTask` and `ClassificationTask`. The following functions are defined for each `SyntheticTask`:
 - `rand(task::SyntheticTask)`: returns a randomly sampled `(input, output)` pair.
 - `rand(task::SyntheticTask, n::Int)`: returns `n` randomly sampled `(input, output)` pair.
 - `size(task::SyntheticTask)`: returns a tuple `(num_input_dims, num_output_dims)`.
 - `size(task::SyntheticTask, 1)`: returns the number of input dimensions.
 - `size(task::SyntheticTask, 2)`: returns the number of output dimensions.
 - `eltype(task::SyntheticTask)`: returns a type `Tuple{input_type, output_type}`.

## Synthetic Tasks

### `AddTask`
Synthetic Addition Task: Given a sequence of vectors `[x[t,1], x[t,2]]` where `x[t,1]` is uniformly distributed in `(0,1)` and `x[t,2] == 1` for exactly two times `t` and `0` for all other times, output the sum of `x[t,1]` where `x[t,2] == 1` . Used in the paper [Orthogonal RNNs and Long-Memory Tasks](http://arxiv.org/abs/1602.06662).

```julia
julia> rand(AddTask(5))
([[0.74,0.0],[0.8,1.0],[0.69,0.0],[0.66,1.0],[0.44,0.0]],1.46)
```

