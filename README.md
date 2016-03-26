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
* nn.TemporalConvolution2  This is specific to clnn.  It works on cpu and cuda too, not just on OpenCL.  It is API-compatible with TemporalConvolution,
and faster than TemporalConvolution, on both CUDA and OpenCL.

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

* have an OpenCL-enabled GPU device available, and appropriate OpenCL-enabled drivers installed

### Procedure

IMPORTANT!  THIS HAS CHANGED!  Please install a specific Torch distro.  In fact it is sufficient to follow
the installation instructions for https://github.com/hughperkins/cltorch , and this will install `clnn`
automatically.

You can check things are working after install by running the unit-tests:
```
luajit -l clnn -e 'clnn.test()'
```

## Updating

* Please do not run `luarocks install torch` or `luarocks install nn`.  This will break your installation
* It is safe to run `luarocks install cltorch` or `luarocks install clnn`
* It's probably best to simply update your distro, ie:
```
cd ~/torch-cl
git pull
git submodule update --recursive
./install.sh
```

## Unit-tests

To run, do:
```
luajit -l clnn -e 'clnn.test()'
```

## Porting guidelines

Porting guidelines, for project maintainers, available here: [porting-guidelines.md](doc/porting-guidelines.md).

## Recent changes

* 2nd May:
  * Re-applied:
    * 26th March:
      * add TemporalConvolution2: same API and usage as TemporalConvolution, but faster on GPUs
* 31st April:
  * Re-applied:
    * 10th March:
      * [@pawni](https://github.com/pawni) (Nick Pawlowski) added SpatialUpSamplingNearest.  Thank you Nick
    * 20th February:
      * [@gloine](https://github.com/gloine) (Jaehyung Lee) added support for non-batched input to ClassNLLCriterion.  Thank you Jaehyung
* 30th April:
  * rolled back to as-of 21st February, prior to lots of THNN changes in upstream Torch
  * additionally, installation procedure is now to use a specific torch distro, for stability
* 1st Feb:
  * merged/ported THNN phase 3.  Any weird build issues, please update both `nn` and `clnn`.
* 2nd January, 2016:
  * merged/ported THNN architecture across, and the implementation of Abs, so the unit-tests pass again now
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

