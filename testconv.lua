require 'clnn'

local net = nn.SpatialConvolutionMM(2,1,3,3,1,1,0,0):cl()
--net.weight:fill(0)
net.bias:fill(0)
--net.weight[1][1] = 1
print('net.weight:size()', net.weight:size())
local input = torch.ClTensor(4,2,5,5):uniform()
print('input[1]', input[1])
print('input[2]', input[2])
local output = net:forward(input)
net.finput:fill(-1)
local output = net:forward(input)
--print('output', output)

print('finput', net.finput)

local netfloat = net:clone():float()
local outfloat = netfloat:forward(input:float())
print('maxdiff', output:float():csub(outfloat):abs():max())
if not (output:float():csub(outfloat):abs():max() <= 0.0001) then
  print('net.weight', net.weight)
  print('netfloat.weight', netfloat.weight)
  print('output:float()', output:float())
  print('outfloat', outfloat)
  print('diff')
  print(output:float():csub(outfloat))
end
assert(output:float():csub(outfloat):abs():max() <= 0.0001)
print('all ok :-)')


