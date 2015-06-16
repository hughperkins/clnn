# clnn

Experimental OpenCL backend for Torch nn neural networks library

Note that the cltorch OpenCL backend for Torch has moved to [https://github.com/hughperkins/cltorch](https://github.com/hughperkins/cltorch)

## What works

Not much so far :-)

- `forward` and `backward` on a `Linear` layer work ok.
- can use stochasticgradientdescent on a linear layer

<pre>
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

