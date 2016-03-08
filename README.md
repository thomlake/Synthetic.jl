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

### `AddTask <: RegressionTask`
Given a sequence of vectors `[x[t,1], x[t,2]]` where `x[t,1]` is uniformly distributed in `(0,1)` and `x[t,2] == 1` for exactly two times `t` and `0` for all other times, output the sum of `x[t,1]` where `x[t,2] == 1`. This task is described in the paper [Orthogonal RNNs and Long-Memory Tasks](http://arxiv.org/abs/1602.06662).

```julia
julia> rand(Synthetic.AddTask(5))
([[0.74,0.0],[0.8,1.0],[0.69,0.0],[0.66,1.0],[0.44,0.0]],1.46)
```

### `CopyTask <: ClassificationTask`
Let `Σ = {1, 2, ..., K}` be an alphabet of `K` symbols, `b=K+1` represent a blank symbol, and `f=K+2` a flag symbol. The input consists of `n` symbols in `Σ`, followed by `m-1` blank symbols, the flag symbol, and then `n` blank symbols. The output is `n + m` blank symbols followed by the first `n` symbols in the input sequence. This task is described in the paper [Orthogonal RNNs and Long-Memory Tasks](http://arxiv.org/abs/1602.06662).

```julia
julia> m, n, K = 4, 3, 2 # delay, capacity, |Σ| + |{b,f}|
julia> rand(Synthetic.CopyTask(m, n, K))
([1,2,1,3,3,3,4,3,3,3],[3,3,3,3,3,3,3,1,2,1])
```

### `XORTask <: ClassificationTask`
The input consists of a random sequence of `1`s and `2`s. The output at each time `t` is given by `(cumsum(input[1:t] - 1) % 2) + 1`.

```julia
julia> x, y =rand(Synthetic.XORTask(10))
([1,1,2,2,1,1,1,1,2,1],[1,1,2,1,1,1,1,1,2,2])
```

