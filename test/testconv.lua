require 'clnn'

cltorch.setDevice(cltorch.getDeviceCount())

local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 1
local times = {}

local from = math.random(1,32)
local to = math.random(1,8) * 8
local ki = math.random(3,15)
local kj = math.random(3,15)
local si = 1 -- not supported by CPU version yet
local sj = si
local outi = math.random(1,64)
local outj = math.random(1,64)
local ini = (outi-1)*si+ki
local inj = (outj-1)*sj+kj

local tm = {}
local title = string.format('SpatialConvolutionMM.backward %dx%dx%d o %dx%d -> %dx%dx%d',
                           from, inj, ini, kj, ki, to, outj, outi)
times[title] = tm

local input = torch.randn(from,inj,ini)
local gradOutput = torch.randn(to,outj,outi)
local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj)
sconv:forward(input)
sconv:zeroGradParameters()
local groundgrad = sconv:backward(input, gradOutput)
local a = torch.Timer()
for i = 1,nloop do
  sconv:zeroGradParameters()
  groundgrad = sconv:backward(input, gradOutput)
end
local groundweight = sconv.gradWeight
local groundbias = sconv.gradBias
tm.cpu = a:time().real

input = input:cl()
gradOutput = gradOutput:cl()
local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cl()
gconv.weight = sconv.weight:cl()
gconv.bias = sconv.bias:cl()
gconv:forward(input)
gconv:zeroGradParameters()
local rescl = gconv:backward(input, gradOutput)
a:reset()
for i = 1,nloop do
  gconv:zeroGradParameters()
  rescl = gconv:backward(input, gradOutput)
end
local weightcl = gconv.gradWeight
local biascl = gconv.gradBias
cltorch.synchronize()
tm.gpu = a:time().real

local error = rescl:float() - groundgrad
local werror = weightcl:float() - groundweight
local berror = biascl:float() - groundbias


