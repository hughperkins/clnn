local _test = clnn._test
local times = _test.times
local clnntest = _test.clnntest
local x_clnntest = _test.x_clnntest
local nloop = _test.nloop
local precision_forward = 1e-4
local precision_backward = 1e-2

function clnntest.SpatialConvolutionMM_forward_single()
   torch.manualSeed(123)
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
   local title = string.format('SpatialConvolutionMM.forward %dx%dx%d o %dx%d -> %dx%dx%d [s: %dx%d]',
      from, inj, ini, kj, ki, to, outj, outi, sj, si)
   times[title] = tm
   
   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cl()
   gconv.weight = sconv.weight:cl()
   gconv.bias = sconv.bias:cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), 1e-5, 'error on state (forward) ')
end

function clnntest.SpatialConvolutionMM_forward_single_padded()
   torch.manualSeed(123)
   local from = 27
   local to = 5
   local ki = 3 -- VGG
   local kj = 3
   local si = 1
   local sj = si
   local outi = 31
   local outj = 25
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local tm = {}
   local title = string.format('SpatialConvolutionMM.forward %dx%dx%d o %dx%d -> %dx%dx%d [s: %dx%d]',
      from, inj, ini, kj, ki, to, outj, outi, sj, si)
   times[title] = tm
   
   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,1,1)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,1,1):cl()
   gconv.weight = sconv.weight:cl()
   gconv.bias = sconv.bias:cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundtruth
   local err_max = error:abs():max()
   local val_max = groundtruth:max()
   local val_abs_avg = groundtruth:clone():abs():mean()
--   print('err_max', err_max, 'val_max', val_max, 'val_avg', val_abs_avg)
   mytester:assertlt(error:abs():max(), 1e-5, 'error on state (forward) ')
end

function clnntest.SpatialConvolutionMM_forward_batch()
   torch.manualSeed(123)
   local bs = math.random(1,4) * 4
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
   local title = string.format('SpatialConvolutionMM.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]',
      bs, from, inj, ini, kj, ki, bs, to, outj, outi, sj, si)
   times[title] = tm
   
   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cl()
   gconv.weight = sconv.weight:cl()
   gconv.bias = sconv.bias:cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), 1e-5, 'error on state (forward) ')
end

function clnntest.SpatialConvolutionMM_backward_single()
   torch.manualSeed(123)
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
   
   mytester:assertlt(error:abs():max(), 1e-5, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), 1e-4, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), 1e-4, 'error on bias (backward) ')
end

function clnntest.SpatialConvolutionMM_backward_batch()
   torch.manualSeed(123)
   local bs = math.random(1,4) * 4
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
   local title = string.format('SpatialConvolutionMM.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
      bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm
   
   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
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
   
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

