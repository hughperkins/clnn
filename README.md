# clnn

Experimental OpenCL backend for Torch nn neural networks library

Note that the cltorch OpenCL backend for Torch has moved to [https://github.com/hughperkins/cltorch](https://github.com/hughperkins/cltorch)

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
* nn.SpatialMaxPooling (note: stride must match pooling size, for now)

### Transfer function layers

* nn.Tanh
* nn.Sigmoid
* nn.ReLU
* nn.LogSoftMax

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

### Trainers

Trainers 'just work', since they just call standard methods on the network to be trained.  Tested with:
* nn.StochasticGradient

# Timings

## Mnist

Using the network in [test/test-mnist2.lua](test/test-mnist2.lua), and `MODEL=conv1`, following timings using an NVidia 940M, per epoch:
* `API=cuda`: 3.2 seconds
* `API=cl`: 13.6 seconds

Note that this network is a bit unfair on clnn, since these are really tiny layers and inputs, for which clnn does less well currently, see the table in 'Soumith benchmark layers', below.

(hmmm, interestingly, on this tiny network, [DeepCL](https://github.com/hughperkins/DeepCL) is actually faster than both.  2.3 seconds per epoch, using `./train numtrain=5120 numtest=-1 netdef=32c5-tanh-mp3-64c5-tanh-mp2-200n-tanh-10n`.)

## Soumith benchmark layers

On an NVidia 940M, using [test/test-perf.lua](test/test-perf.lua):

| layer | direction | cuda time (seconds) | cl time (seconds) |
|-------|-----------|---------------------|----------------|
| l1    | forward   | 1.02               | 1.14    |
| l2    | forward   | out of mem          | out of mem     |
| l3    | forward   | 0.85                | 1.19 |
| l4    | forward   | 0.15                | 0.42 |
| l5    | forward   | 0.22                | 0.37 |

| layer | direction | cuda time (seconds) | cl time (seconds) |
|-------|-----------|---------------------|----------------|
| l1    | backward  | 0.93+1.47 =2.4              | 1.25+1.43 = 2.68    |
| l2    | backward   | didnt try          | didnt try    |
| l3    | backward   | 0.84+0.64 =1.48                | 0.93+2.28=3.21 |
| l4    | backward   | 0.11+0.11 =0.22               | 0.17+0.20=0.37 |
| l5    | backward   | 0.13+0.16=0.29                | 0.23+0.91=1.14 |

## Example network

* Here is an OpenCL-enabled version of Karpathy's LSTM network: [https://github.com/hughperkins/char-rnn](https://github.com/hughperkins/char-rnn)
* Simply add option `-opencl 1` to enable OpenCL :-)

## Installation

### Pre-requisites

* have installed:
  * [torch](https://github.com/torch/torch7)
  * [nn](https://github.com/torch/nn)
  * [cltorch](https://github.com/hughperkins/cltorch)
* have an OpenCL-enabled GPU device available, and appropriate OpenCL-enabled drivers installed

### Procedure

```
git clone https://github.com/hughperkins/clnn.git
cd clnn
luarocks make rocks/clnn-scm-1.rockspec
```

You should now be able to use `require 'clnn'` from your lua scripts :-)

# Unit-tests

* For all layers except SpatialConvolutionMM, please see:
  * [test/test-layers.lua](test/test-layers.lua)
* For SpatialConvolutionMM, please see:
  * [test/test-spatialconvolution.lua](test/test-spatialconvolution.lua) (Needs `cunn` available, to do numerical comparison)

# Porting guidelines

Porting guidelines, for project maintainers, available here: [porting-guidelines.md](doc/porting-guidelines.md).

# Recent changes

* 27th June:
  * mild perf improvement to LogSoftMax layer
  * removed FullyConnected for now
  * mild perf improvement to Narrow layer
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

