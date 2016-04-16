-- simply try multiplying different sized matrics
-- as though it is alexnet layer 1

require 'clnn'

for b=1,3 do
  local groupSize = 16

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

  local outW = 55
  local outH = 55

  local columns = torch.ClTensor(groupSize, outPlanes, kW * kH * inPlanes)
  local weights = torch.ClTensor(kW * kH * inPlanes, outW * outH)
  local output = torch.ClTensor(groupSize, outPlanes, outW * outH)

  cltorch.synchronize()
  sys.tic()
  for it=1,10  do
    for i=1,16 do
      output[i]:mm(columns[i], weights)
    end
  end
  cltorch.synchronize()
  print(sys.toc())

  local columns = torch.ClTensor(groupSize * outPlanes, kW * kH * inPlanes)
  local weights = torch.ClTensor(kW * kH * inPlanes, outW * outH)
  local output = torch.ClTensor(groupSize * outPlanes, outW * outH)

  cltorch.synchronize()
  sys.tic()
  for it=1,10  do
    output:mm(columns, weights)
  end
  cltorch.synchronize()
  print(sys.toc())
--  print('output:size()', output:size())
end

