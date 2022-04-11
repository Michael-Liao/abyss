# Abyss
> a multi-dimensional array library that focuses on GPU computation.
Currently only works with CPU.
CUDA will be supported in the future.

## Quick Start

## Prerequisites
cmake >= 3.15
ninja
openblas ?

## Installation
In the project directory
```
mkdir build && cd build

# generate build files for your favorite build tool
cmake ..

# build
ninja
ninja install
```

## Roadmap
[x] Typeless Tensor
[ ] Vector operations
[ ] Broadcasting
[ ] Slicing
[ ] Matrix operations
[ ] I/O
[ ] CUDA backend