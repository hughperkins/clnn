require 'nn'
require 'cltorch'
require 'clnn'

local layer = nn.FullyConnected(10)
local input = torch.Tensor(2, 4, 3, 3):uniform()
print('input\n', input)

local output = layer:forward(input)
print('output\n', output)

layer:zeroGradParameters()
layer:backward(input, output)
print('gradInput\n', layer.gradInput)
print('gradWeight\n', layer.gradWeight)
print('gradBias\n', layer.gradBias)


