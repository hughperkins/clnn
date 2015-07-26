require 'clnn'

local n = nn.Apply(3, 2, [[
  {{out1}} = {{in1}} + {{in2}};
  {{out2}} = {{in3}} + 3.0f;
]], [[
  {{in1}} = {{out1}};
  {{in2}} = {{out1}};
  {{in3}} = {{out2}};
]])

local in1 = torch.ClTensor(3,2):uniform()
local in2 = torch.ClTensor(3,2):uniform()
local in3 = torch.ClTensor(3,2):uniform()
local inputs = {in1, in2, in3}

local outputs = n:forward(inputs)
print('in1', in1)
print('in2', in2)
print('in3', in3)
print('outputs\n', outputs, outputs[1], outputs[2])

local gradInput = n:backward(inputs, outputs)
print('gradInput\n', gradInput, gradInput[1], gradInput[2], gradInput[3])

