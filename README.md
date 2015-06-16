# clnn

Experimental OpenCL backend for Torch nn neural networks library

Note that the cltorch OpenCL backend for Torch has moved to [https://github.com/hughperkins/cltorch](https://github.com/hughperkins/cltorch)

## What works

Not much so far :-)

Basically, `forward` and `backward` on a `Linear` layer work ok.  That's it for now ;-)  Cannot train :-P

<pre>
l1cl = nn.Linear(3, 2):cl()
C = torch.ClTensor{3,5,2}
print('l1cl:forward(A)\n', l1cl:forward(C))

gradOutputCl = torch.ClTensor{0.5, -0.8}
print(l1cl:backward(C, gradOutputCl))
</pre>

