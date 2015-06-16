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

l1cl = l1:clone():cl()
print('l1cl\n', l1cl)
outcl = l1cl:forward(C)
print('out\n', outcl)

print('l1.weight\n', l1.weight)
print('l1cl.weight\n', l1cl.weight)

function dump(object)
  for k,v in pairs(object) do
    print(k,v)
  end
end

gradOutput = torch.Tensor{0.5, -0.8}
a = l1:backward(A, gradOutput)
print('a\n', a)

gradOutputCl = gradOutput:cl()
c = l1cl:backward(C, gradOutputCl)
print('c\n', c)

l1 = nn.Linear(1,1)
l1cl = l1:clone():cl()
dataset = {}
e1 = {}
e1[1] = torch.Tensor{0}
e1[2] = torch.Tensor{1}
e2 = {}
e2[1] = torch.Tensor{1}
e2[2] = torch.Tensor{3}
dataset[#dataset+1] = e1
dataset[#dataset+1] = e2
function dataset.size()
  return 2
end

print('e1[1].nn', e1[1].nn)

criterion = nn.MSECriterion() -- Mean Squared Error criterion
trainer = nn.StochasticGradient(l1, criterion)
trainer.learningRate = 0.1  
print('learningRate', trainer.learningRate)
trainer:train(dataset) -- train using some examples

print('e1[1].nn', e1[1].nn)
print(l1:forward(e1[1]))
print(l1:forward(e2[1]))
print('e1[1].nn', e1[1].nn)
-- dump(e1[1].nn)

print(torch.Tensor({3,5}).nn)

print('criterion\n', criterion)
datasetcl = {}
for i,e in ipairs(dataset) do
  print('e[1]', e[1])
  print('e[2]', e[2])
  ecl = {}
  ecl[1] = e[1]:clone():cl()
  ecl[2] = e[2]:clone():cl()
  print('ecl[1]', ecl[1])
  print('ecl[2]', ecl[2])
--  vcl = {}
--  vcl[1] = v[1]:clone():cl()
--  vcl[2] = v[2]:clone():cl()
  datasetcl[#datasetcl+1] = ecl
end
function datasetcl.size()
  return 2
end
trainercl = nn.StochasticGradient(l1cl, criterion:clone():cl())
trainercl.learningRate = 0.1  
print('learningRate', trainercl.learningRate)
trainercl:train(datasetcl) -- train using some examples

print(l1cl:forward(datasetcl[1][1]))
print(l1cl:forward(datasetcl[2][1]))

