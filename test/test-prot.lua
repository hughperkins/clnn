require 'nn'

input = torch.Tensor(128,32,28,28):uniform()

layer = nn.LogSoftMax()

output = layer:forward(input)

print('output:size()', output:size())

