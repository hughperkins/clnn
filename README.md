# clnn

Experimental OpenCL backend for Torch nn neural networks library

Note that the cltorch OpenCL backend for Torch has moved to [https://github.com/hughperkins/cltorch](https://github.com/hughperkins/cltorch)

## What works

*Containers:*
I suppose all containers work unchanged.  Tested however so far on:
* nn.Sequential

*Weighted layers:*
* nn.Linear (unchanged, since uses matrix operations on whatever tensors we feed it)
* nn.SpatialConvolutionMM (not tested for correctness yet)

*Pooling layers*
* nn.SpatialMaxPooling (not tested for correctness yet)

*Activation layers:*
* nn.Tanh
* nn.Sigmoid
* nn.ReLU

*Criterion:*
* nn.MSECriterion

*Trainers:*
I suppose all trainers work unchanged.  Tested however so far using:
* nn.StochasticGradient

# Samples

* For training on mnist, you can run `./run-mnist2.sh`
* Options:
  * Use env var `API` to choose `cpu`, `cuda`, or `cl`
  * Use env var `MODEL` to choose `linear` or `conv1`
* eg run like this:
```
API=cl MODEL=conv1 ./run-mnist2.sh
```

# Timings

## Mnist

Using the network in [test/test-mnist2.lua](test-mnist2.lua), and `MODEL=conv1`, following timings using an NVidia 940M, per epoch:
* `API=cuda`: 3.2 seconds
* `API=cl`: 13.6 seconds

(hmmm, interestingly, on this tiny network, DeepCL is actually faster than both.  2.3 seconds per epoch, using `./train numtrain=5120 numtest=-1 netdef=32c5-tanh-mp3-64c5-tanh-mp2-200n-tanh-10n`.)

## Soumith benchmark layers

On an Nvidia 940M, using [test/test-perf.lua](test/test-perf.lua):

| layer | direction | cuda time (seconds) | cl time (seconds) |
|-------|-----------|---------------------|----------------|
| l1    | forward   | 1.017               | 1.14    |
| l2    | forward   | out of mem          | out of mem     |
| l3    | forward   | 0.85                | 1.19 |
| l4    | forward   | 0.148                | 0.42 |
| l5    | forward   | 0.22                | 0.37 |

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

# Porting guidelines

Porting guidelines, for project maintainers, available here: [porting-guidelines.md](doc/porting-guidelines.md).


