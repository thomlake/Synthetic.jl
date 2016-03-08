# Synthetic.jl
Synthetic.jl is a Julia package for generating synthetic data. It is designed to be used for testing and sanity checking Machine Learning algorithms.

## Overview
Synthetic.jl defines an abstract type `SyntheticTask` and two abstract  subtypes: `RegressionTask` and `ClassificationTask`. The following functions are defined for each `SyntheticTask`:
 - `rand(task)`: returns a randomly sampled (input, output) pair.
 - `rand(task, n::Int)`: returns `n` randomly sampled (input, output) pair.
 - `size(task)`: returns a tuple `(num_input_dims, num_output_dims)`.
 - `size(task, 1)`: returns the number of input dimensions.
 - `size(task, 2)`: returns the number of output dimensions.
 - `eltype(task)`: returns a type `Tuple{input_type, output_type}`.

## Synthetic Tasks
