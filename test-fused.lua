require 'clnn'

local n = nn.Apply(3, 2, [[
  {{out1}} = {{in1}} + {{in2}};
  {{out2}} = {{in3}} + 3.0f;
]])

local in1 = torch.ClTensor(3,2):uniform()
local in2 = torch.ClTensor(3,2):uniform()
local in3 = torch.ClTensor(3,2):uniform()

local output = n:forward({in1, in2, in3})
print('in1', in1)
print('in2', in2)
print('in3', in3)
print('output\n', output, output[1], output[2])

