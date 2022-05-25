# Abyss
> a deep learning library that focuses on education and transparency.

The main goal of this project is to learn and possibly teach how deep learning libraries are implemented. The main objectives are:

* Learning how to use modern CMake for cross-platform builds
* Implementation of a sound numpy-like n-dimensional array.
* Implementation of Computational Graph
* Understand the pytorch API and why is it created this way.
* Train MNIST on a simple model

Currently only works with CPU.
CUDA will be supported in the future.

## Quick Start

## Prerequisites
cmake >= 3.15
ninja
openblas 0.3.13
libjpeg-turbo 2.0.6 (libturbojpeg0-dev on ubuntu)
libpng 1.6.37

### optional dependencies
Catch2 2.13.9
doxygen 1.18
CUDA 11

## Installation
In the project directory
```
mkdir build

# generate build files for your favorite build tool
cmake -G "Ninja" -B ./build

# build
cmake --build ./build
cmake --install ./build # may require sudo priveleges
```

## Roadmap
- [x] Typeless Tensor
- [x] Vector operations
- [x] Broadcasting
- [x] Slicing
- [x] Matrix multiplication
- [x] Computational graph implementation and back-propagation
- [ ] NN model interface (`torch.nn.Module`)
- [ ] optimizers (`torch.optim`)
- [ ] I/O functions and data loaders (like `torch.util.data.Dataset` and `torch.util.data.DataLoader`)
- [ ] CUDA backend