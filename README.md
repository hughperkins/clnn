# clnn

Experimental OpenCL backend for Torch nn neural networks library

Note that the cltorch OpenCL backend for Torch has moved to [https://github.com/hughperkins/cltorch](https://github.com/hughperkins/cltorch)

## What works

*Containers:*

I suppose all containers work unchanged.  Tested however so far on:
* nn.Sequential

*Weighted layers:*
* nn.Linear (unchanged, since uses matrix operations on whatever tensors we feed it)
* nn.SpatialConvolutionMM

*Pooling layers*
* nn.SpatialMaxPooling

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

Using the network in [test/test-mnist2.lua](test/test-mnist2.lua), and `MODEL=conv1`, following timings using an NVidia 940M, per epoch:
* `API=cuda`: 3.2 seconds
* `API=cl`: 13.6 seconds

(hmmm, interestingly, on this tiny network, DeepCL is actually faster than both.  2.3 seconds per epoch, using `./train numtrain=5120 numtest=-1 netdef=32c5-tanh-mp3-64c5-tanh-mp2-200n-tanh-10n`.)

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

# Recent changes

* 22nd June:
  * Checked that SpatialConvolutionMM gives same results using clnn, compared with cunn
  * Checked that SpatialMaxPooling gives same results using clnn, compared with nn
  * -Created nn.FullyConnected layer type, which handles Reshape for us :-)- (note: rethinking this...)
  * Added ReLU, which was already marked as added but ... wasnt :-P  but now is :-) )
* 21st June:
  * Got SpatialConvolutionMM and SpatialMaxPooling running
  * Ran Soumith benchmarks on SpatialConvolutionMM, for clnn and cunn, on NVidia 940M

