# clnn

OpenCL backend for Torch nn neural networks library.

## What works

### Parameterized Modules

* nn.Linear

### Basic Tensor methods

These mostly 'just work', since based on underlying tensor methods, already implemented in [cltorch](https://github.com/hughperkins/cltorch).  Tested with:

* nn.Narrow

### Miscellaneous modules

* nn.Identity
* nn.Dropout

### Convolution layers

* nn.SpatialConvolutionMM
* nn.SpatialMaxPooling (including `ceil` mode)
* nn.SpatialAveragePooling

### Transfer function layers

* nn.Tanh
* nn.Sigmoid
* nn.ReLU
* nn.ELU
* nn.Exp
* nn.Sqrt
* nn.Square
* nn.Abs
* nn.LogSigmoid
* nn.HardTanh
* nn.LogSoftMax
* nn.SoftMax (including spatial mode)

### Table layers

These 'just work', since they are based on underlying torch operations, which are already implemented in [cltorch](https://github.com/hughperkins/cltorch).  Tested with:
* nn.CMulTable
* nn.CAddTable

### Criterions

* nn.MSECriterion
* nn.ClassNLLCriterion

### Containers:

Containers 'just work', since they just call standard operations on the contained modules.  Tested with:
* nn.Sequential
* nngraph

### Trainers

In theory, trainers 'just work', since they just call standard torch methods on the network.  The following are good first choices:
* nn.StochasticGradient
* optim.lbfgs
* optim.adam

## Timings

### Soumith benchmark layers

Please see https://github.com/soumith/convnet-benchmarks#imagenet-winners-benchmarking
* On a Titan X, OpenCL torch is about 3 times slower than CUDA torch
  * eg for VGG, cutorch takes 1100ms, and cltorch takes 3400ms

## Example networks

* Andrej's [char-rnn](https://github.com/karpathy/char-rnn) is OpenCL-enabled, simple add option `-opencl 1`
* Justin's [neural-style](https://github.com/jcjohnson/neural-style) has an OpenCL port in progress by Shubhanshu [napsternxg/neural-style](https://github.com/napsternxg/neural-style)

## Installation

### Pre-requisites

* have installed:
  * [torch](https://github.com/torch/torch7)
  * [nn](https://github.com/torch/nn)
  * [cltorch](https://github.com/hughperkins/cltorch)
* have updated, right now, cltorch, to latest version, eg `luarocks install cltorch`
  * any weird build issues on clnn, or seg faults etc, please verify cltorch is latest version before raising issue
* have an OpenCL-enabled GPU device available, and appropriate OpenCL-enabled drivers installed

### Procedure

```
luarocks install clnn
```

You should now be able to use `require 'clnn'` from your lua scripts :-)

Please check that all is working by running the unit-tests:
```
luajit -l clnn -e 'clnn.test()'
```

## Updating

* Please update to latest version of cltorch before updating to latest version of clnn
* If you update cltorch, please afterwards also update clnn

## Unit-tests

To run, do:
```
luajit -l clnn -e 'clnn.test()'
```

## On the bleeding edge: getting latest SpatialAveragePooling

* latest SpatialAveragePooling has been ported from Sergey's [SpatialAveragePadding and ceil kernels](https://github.com/torch/cunn/pull/134)
* you need to update your `nn` first, to a post-master fork:
```
git clone https://github.com/hughperkins/nn.git -b avepool_plus_master nn-avepool
cd nn-avepool
luarocks make rocks/nn-scm-1.rockspec
cd ..
```
* now, you can update `clnn` to a post-master fork:
```
git clone https://github.com/hughperkins/clnn.git -b avgpool clnn-avgpool
cd clnn-avgpool
luarocks make rocks/clnn-scm-1.rockspec
cd ..
```
* finally, run tests:
```
luajit -l clnn -e 'clnn.test()'
```
* you can see an example of using it in Justin's [neural-style](https://github.com/jcjohnson/neural-style), in the [OpenCL support](https://github.com/jcjohnson/neural-style/issues/44#issuecomment-142912267) issue

## Porting guidelines

Porting guidelines, for project maintainers, available here: [porting-guidelines.md](doc/porting-guidelines.md).

## Recent changes

* 15th December:
  * merged Sergey's [SpatialAveragePadding and ceil kernels](https://github.com/torch/cunn/pull/134) into `master` branch
* 29th November:
  * added ELU
* 25th September:
  * ported Sergey's not-yet-merged  [SpatialAveragePadding and ceil kernels](https://github.com/torch/cunn/pull/134), into `clnn-avgpool` branch
  * ported latest version of SoftMax, ie essentially Jonghoon's [Update SoftMax to work in spatial mode](https://github.com/torch/cunn/pull/135)
* 23rd September:
  * ported latest cunn implementation of `SpatialMaxPooling` across, ie approximately Sergey's [Deterministic max-pooling](https://github.com/torch/cunn/pull/106) PR
    * this includes `:ceil()` implementation
* 22nd September:
  * added non-batch implementation of LogSoftMax (previously only handled batched input)
  * added SoftMax, for both batched and non-batched
* 20th September:
  * added non-batch implementation for SpatialMaxPooling (previously only handled batched input), for contiguous pools

[Older changes](doc/older-changes.md)

