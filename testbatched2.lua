-- simply try multiplying different sized matrics
-- as though it is alexnet layer 1

require 'clnn'

local groupSize = 16

--local inPlanes = 3
--local outPlanes = 64
--local inW = 224
--local inH = 224
--local kW = 11
--local kH = 11
--local strideW = 4
--local strideH = 4
--local padW = 2
--local padH = 2

--local outW = 55
--local outH = 55

--0.46545720100403
--0.1340479850769


--local inPlanes = 64
--local outPlanes = 192
--local inW = 55
--local inH = 55
--local kW = 5
--local kH = 5
--local strideW = 1
--local strideH = 1
--local padW = 1
--local padH = 1

--local outW = 27
--local outH = 27

--0.55227398872375
--0.23579287528992


--local inPlanes = 192
--local outPlanes = 384
--local inW = 13
--local inH = 13
--local kW = 3
--local kH = 3
--local strideW = 1
--local strideH = 1
--local padW = 1
--local padH = 1

--local outW = 13
--local outH = 13

--0.30923318862915
--0.30467700958252


--local inPlanes = 384
--local outPlanes = 256
--local inW = 13
--local inH = 13
--local kW = 3
--local kH = 3
--local strideW = 1
--local strideH = 1
--local padW = 1
--local padH = 1

--local outW = 13
--local outH = 13

--0.40928387641907
--0.40597796440125


local inPlanes = 256
local outPlanes = 256
local inW = 13
local inH = 13
local kW = 3
local kH = 3
local strideW = 1
local strideH = 1
local padW = 1
local padH = 1

local outW = 13
local outH = 13

--0.27404594421387
--0.27145791053772

  local columns = torch.ClTensor(groupSize, outPlanes, kW * kH * inPlanes)
  local weights = torch.ClTensor(kW * kH * inPlanes, outW * outH)
  local output = torch.ClTensor(groupSize, outPlanes, outW * outH)

for b=1,3 do
  cltorch.synchronize()
  sys.tic()
  for it=1,10  do
    for i=1,16 do
      output[i]:mm(columns[i], weights)
    end
  end
  cltorch.synchronize()
  print(sys.toc())
end

  local columns = torch.ClTensor(groupSize * outPlanes, kW * kH * inPlanes)
  local weights = torch.ClTensor(kW * kH * inPlanes, outW * outH)
  local output = torch.ClTensor(groupSize * outPlanes, outW * outH)

for b=1,3 do
  cltorch.synchronize()
  sys.tic()
  for it=1,10  do
    output:mm(columns, weights)
  end
  cltorch.synchronize()
  print(sys.toc())
end
--  print('output:size()', output:size())

