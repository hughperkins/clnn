

require 'clnn'

-- we will simulate the first alexnet layer, and simulate 16 image of 3x224x224, with 64 output planes
-- or one image of 3x224 with 16*64 = 1024 output planes

-- method 1

local inPlanes = 3
local outPlanes = 64
local inW = 224
local inH = 224
local kW = 11
local kH = 11
local strideW = 4
local strideH = 4
local padW = 2
local padH = 2

local net = nn.SpatialConvolutionMM(inPlanes, outPlanes, kW, kH, strideW, strideH, padW, padH):cl()
local inputs = torch.ClTensor(1, inPlanes, inW, inH):uniform()

times = {}
for it=1,3 do
  sys.tic()
  cltorch.synchronize()
  for b=1,16 do
    local out = net:forward(inputs)
    local gradInput = net:backward(inputs, out)
  end
  cltorch.synchronize()
  table.insert(times, sys.toc())
--  print(sys.toc())
end

--m=64 k=363 n=3025
--columns
--[torch.ClTensor of size 363x3025]
--weight
--[torch.ClTensor of size 64x363]
--output_n
--[torch.ClTensor of size 64x55x55]
--input size 1 3 224 224 
--columns: 363 * 3025
--output: 1 x 64 x 55 x 55
--  batch=0/1
--m=64 k=363 n=3025

for i, time in ipairs(times) do
  print(i, time)
end

-- now try method 2 ...

local inPlanes = 3
local outPlanes = 64 * 16
local inW = 224
local inH = 224
local kW = 11
local kH = 11
local strideW = 4
local strideH = 4
local padW = 2
local padH = 2

local net = nn.SpatialConvolutionMM(inPlanes, outPlanes, kW, kH, strideW, strideH, padW, padH):cl()
local inputs = torch.ClTensor(1, inPlanes, inW, inH):uniform()

times = {}
for it=1,3 do
  sys.tic()
  cltorch.synchronize()
  local out = net:forward(inputs)
  local gradInput = net:backward(inputs, out)
  cltorch.synchronize()
  table.insert(times, sys.toc())
--  print(sys.toc())
end

for i, time in ipairs(times) do
  print(i, time)
end

