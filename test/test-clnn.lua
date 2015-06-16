require 'nn'
require 'clnn'

l1 = nn.Linear(3, 2)
print('l1.weight\n', l1.weight)
print('l1.bias\n', l1.bias)
print('l1.output\n', l1.output)
l1.weight = torch.Tensor{{0.2, -0.2, 0.3},
                        {0.4,-0.1, -0.5}}
l1.bias = torch.Tensor{0.1, -0.2}
print('l1.weight\n', l1.weight)
print('l1.bias\n', l1.bias)

A = torch.Tensor{3,5,2}
print('l1:forward(A)\n', l1:forward(A))
print('l1.output\n', l1.output)
--for k,v in pairs(l1) do
--   print(k,v)
--end

C = A:cl()

--l1.weight = l1.weight:cl()
--l1.bias = l1.bias:cl()
--l1.output = l1.output:cl()
--print('l1.weight\n', l1.weight)
--print('l1.bias\n', l1.bias)
--print('C\n', C)
--out = l1:forward(C)
--print('out\n', out)

l1cl = l1:cl()
print('l1cl\n', l1cl)
outcl = l1cl:forward(C)
print('out\n', outcl)


