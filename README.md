# clnn

Experimental OpenCL backend for Torch nn neural networks library

Note that the cltorch OpenCL backend for Torch has moved to [https://github.com/hughperkins/cltorch](https://github.com/hughperkins/cltorch)

## What works

Layer types:
* nn.Linear
* nn.Tanh

Criterion:
* nn.MSECriterion

Trainers:
* nn.StochasticGradient

# Samples

* For using a Linear layer on mnist, see [test/test-mnist2.lua](test/test-mnist2.lua)  (can run it by doing `API=cl ./run-mnist2.sh`.  Interchange `cl` for `cuda` or `cpu` to compare with cpu and cuda.  To save you the suspense, for some reason cuda is about ten times faster for now....)

## Installation

### Pre-requisites

* have installed [torch](https://github.com/torch/torch7), [nn](https://github.com/torch/nn), [cltorch](https://github.com/hughperkins/cltorch)
* have activated the torch installation

### Procedure

```
git clone https://github.com/hughperkins/clnn.git
cd clnn
luarocks make rocks/clnn-scm-1.rockspec
```

You should now be able to use `require 'clnn'` from your lua scripts :-)

