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
* nn.SpatialAveragePooling (either filter size must equal input size, or filter size must equal stride size)

### Transfer function layers

* nn.Tanh
* nn.Sigmoid
* nn.ReLU
* nn.Exp
* nn.Sqrt
* nn.Square
* nn.Abs
* nn.LogSigmoid
* nn.HardTanh
* nn.LogSoftMax
* nn.SoftMax

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

The source-code for the tests:
* For all layers except SpatialConvolutionMM, please see:
  * [test/test-layers.lua](test/test-layers.lua)
* For SpatialConvolutionMM, please see:
  * [test/test-spatialconvolution.lua](test/test-spatialconvolution.lua) (Needs `cunn` available, to do numerical comparison)

## Porting guidelines

Porting guidelines, for project maintainers, available here: [porting-guidelines.md](doc/porting-guidelines.md).

## Recent changes

* 23rd September:
  * ported latest cunn implementation of `SpatialMaxPooling` across, ie approximately Sergey's [Deterministic max-pooling](https://github.com/torch/cunn/pull/106) PR
    * this includes `:ceil()` implementation
* 22nd September:
  * added non-batch implementation of LogSoftMax (previously only handled batched input)
  * added SoftMax, for both batched and non-batched
* 20th September:
  * added non-batch implementation for SpatialMaxPooling (previously only handled batched input), for contiguous pools
* 10th August:
  * Improve error message when out of memory, ie will say it ran out of memory, rather than say 'c++ exception' now, in many common cases
  * SpatialMaxPooling can now handle pooling size and stride are different, as long as half the pooling size is no more than stride
  * Added SpatialAveragePooling for case where input size equals filter size, or filter size equals stride size
* 22nd July:
  * Performance improvements in underlying [cltorch](https://github.com/hughperkins/cltorch) mean that times for [char-rnn](http://github.com/karpathy/char-rnn) are now around 2-3 times faster on NVIDIA and AMD GPUs
* 6th July:
  * lots of new activations added: `Sqrt`, `Square`, `Exp`, `Abs`, `LogSigmoid`, `HardTanh`  (provided by Sergey Zagoruyko)
  * SpatialMaxPooling:
    * added implicit floor max pooling (provided by Sergey)
    * added 3d forward (from Sergey)
  * added tests from cunn (thank you Sergey)
  * bug fixes:
    * SpatialConvolutionMM updated to match current nn (Sergey)
    * fixed bug in ReLU for in-place forward
* 27th June:
  * mild perf improvement to LogSoftMax layer
  * removed FullyConnected for now
  * mild perf improvement to Narrow layer
  * huge perf improvement :-)  Please update to latest version of [cltorch](http://github.com/hughperkins/cltorch) (should be at least commit 2f1e3e758fb or later)
* 26th June:
  * fixed bug in Sigmoid, which wasnt resizing correctly
* 25th June:
  * added tests for CMulTable and CAddTable, which pass
  * added test for Narrow, which passes
  * fix bug in cmakelists.txt, which meant that installation didnt work (it ran ok for me, so I didnt notice...)
  * Dropout working now
* 24th June:
  * Added ClassNLLCriterion layer (and unit tests for this)
* 23rd June:
  * Added LogSoftMax layer (and unit test for this)
* 22nd June:
  * Checked that SpatialConvolutionMM gives same results using clnn, compared with cunn
  * Checked that SpatialMaxPooling gives same results using clnn, compared with nn
  * Added ReLU, which was already marked as added but ... wasnt :-P  but now is :-) )
* 21st June:
  * Got SpatialConvolutionMM and SpatialMaxPooling running
  * Ran Soumith benchmarks on SpatialConvolutionMM, for clnn and cunn, on NVidia 940M

