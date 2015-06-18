# clnn

Experimental OpenCL backend for Torch nn neural networks library

Note that the cltorch OpenCL backend for Torch has moved to [https://github.com/hughperkins/cltorch](https://github.com/hughperkins/cltorch)

## What works

- `forward` and `backward` on a `Linear` layer work ok.
- can use stochasticgradientdescent on a linear layer
- can train mnist, see [test/test-mnist2.lua](test/test-mnist2.lua)  (can run it by doing `./run-mnist2.sh`)

<pre>
require 'nn'
require 'clnn'

l1cl = nn.Linear(3, 2):cl()
C = torch.ClTensor{3,5,2}
print('l1cl:forward(A)\n', l1cl:forward(C))

gradOutputCl = torch.ClTensor{0.5, -0.8}
print(l1cl:backward(C, gradOutputCl))

datasetcl = {}
e1 = {}
e1[1] = torch.ClTensor{0}
e1[2] = torch.ClTensor{1}
e2 = {}
e2[1] = torch.ClTensor{1}
e2[2] = torch.ClTensor{3}
datasetcl[#dataset+1] = e1
datasetcl[#dataset+1] = e2
function datasetcl.size()
  return 2
end

trainercl = nn.StochasticGradient(l1cl, nn.MSECriterion():cl())
trainercl.learningRate = 0.1  
trainercl:train(datasetcl)

print(l1cl:forward(datasetcl[1][1]))
print(l1cl:forward(datasetcl[2][1]))

</pre>

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

## Vision

I want to be able to use this to run things like [AlexNet](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf), [Clark and Storkey](http://arxiv.org/abs/1412.3409), and [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

## Why I choose torch

I really like the torch `nn` syntax, and naming.  Actually, I started to use the same naming conventions in my [DeepCL](https://github.com/hughperkins/DeepCL) library.  Finally, I thought, oh, let's just get torch nn working on OpenCL somehow :-)   Or try anyway.

And, lots of the people I really admire are here, in torch.  Soumith is in torch.  Karpathy has used torch for at least one project, ie [char-rnn](https://github.com/karpathy/char-rnn).  Lecun uses it, eg see [Lecun's AMA](http://www.reddit.com/r/MachineLearning/comments/25lnbt/ama_yann_lecun/).

