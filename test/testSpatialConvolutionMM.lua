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

function clnntest.SpatialConvolutionMM_forward_single_vgglayer13()
   torch.manualSeed(123)
   local from = 256
   local to = 256
   local ki = 3
   local kj = 3
   local si = 1
   local sj = 1
   local outi = 8
   local outj = 8
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

local function do_SpatialConvolutionMM_forward(params, weight, bias)
   torch.manualSeed(123)
   local from = params.from
   local to = params.to
   local ki = params.k
   local kj = params.k
   local si = 1
   local sj = 1
   local outi = params.out
   local outj = params.out
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local padding = params.padding
   
   local tm = {}
   local title = string.format('SpatialConvolutionMM.forward %dx%dx%d o %dx%d -> %dx%dx%d [s: %dx%d]',
      from, inj, ini, kj, ki, to, outj, outi, sj, si)
   times[title] = tm
   
   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padding,padding)
   if weight ~= nil then
      sconv.weight:copy(weight)
   end
   if bias ~= nil then
      sconv.bias:copy(bias)
   end
   local groundtruth = sconv:forward(input)
   mytester:assertne(groundtruth, nil)
   mytester:asserteq(groundtruth:ne(groundtruth):sum(), 0)
   local a = torch.Timer()
   groundtruth = sconv:forward(input)
   tm.cpu = a:time().real
   
   input = input:cl()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padding,padding):cl()
   gconv.weight = sconv.weight:cl()
   gconv.bias = sconv.bias:cl()
--   print('input', input)
--   print('gweight', gconv.weight)
--   print('gbias', gconv.bias)
   local rescl = gconv:forward(input)
   a:reset()
   rescl = gconv:forward(input)
   cltorch.synchronize()
   tm.gpu = a:time().real
--   print('groundtruth', groundtruth)
--   print('rescl', rescl)
   
   local error = rescl:float() - groundtruth
   local err_max = error:abs():max()
   local val_max = groundtruth:max()
   local val_abs_avg = groundtruth:clone():abs():mean()
--   print('err_max', err_max, 'val_max', val_max, 'val_avg', val_abs_avg)
   mytester:assertlt(error:abs():max(), 1e-5, 'error on state (forward) ')
end

function clnntest.SpatialConvolutionMM_forward_simple_k1()
   local params = {}
   params.from = 1
   params.to = 1
   params.k = 1
   params.out = 1
   params.padding = 0
   do_SpatialConvolutionMM_forward(params)
end

function clnntest.SpatialConvolutionMM_forward_simple_k1_nobias()
   local params = {}
   params.from = 1
   params.to = 1
   params.k = 1
   params.out = 1
   params.padding = 0
   do_SpatialConvolutionMM_forward(params, nil, torch.FloatTensor{0})
end

function clnntest.SpatialConvolutionMM_forward_simple_k2_nobias()
   local params = {}
   params.from = 1
   params.to = 1
   params.k = 2
   params.out = 1
   params.padding = 0
   do_SpatialConvolutionMM_forward(params, nil, torch.FloatTensor{0})
end

function clnntest.SpatialConvolutionMM_forward_simple_k2_padded_nobias()
   local params = {}
   params.from = 1
   params.to = 1
   params.k = 2
   params.out = 1
   params.padding = 1
   do_SpatialConvolutionMM_forward(params, nil, torch.FloatTensor{0})
end

function clnntest.SpatialConvolutionMM_forward_simple_k3_nobias()
   local params = {}
   params.from = 1
   params.to = 1
   params.k = 3
   params.out = 1
   params.padding = 0
   do_SpatialConvolutionMM_forward(params, nil, torch.FloatTensor{0})
end

function clnntest.SpatialConvolutionMM_forward_simple_k3_padded_nobias()
   local params = {}
   params.from = 1
   params.to = 1
   params.k = 3
   params.out = 1
   params.padding = 1
   do_SpatialConvolutionMM_forward(params, nil, torch.FloatTensor{0})
end

function clnntest.SpatialConvolutionMM_forward_simple_k3_padded()
   local params = {}
   params.from = 1
   params.to = 1
   params.k = 3
   params.out = 1
   params.padding = 1
   do_SpatialConvolutionMM_forward(params)
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

function clnntest.SpatialConvolutionMM_forward_batch_square_odd()
   torch.manualSeed(123)
   local bs = 4
   local from = 16
   local to = 24
   local ki = 3
   local kj = 3
   local si = 1
   local sj = 1
   local outi = 28
   local outj = 28
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

function clnntest.SpatialConvolutionMM_forward_batch_square_even()
   torch.manualSeed(123)
   local bs = 4
   local from = 16
   local to = 24
   local ki = 2
   local kj = 2
   local si = 1
   local sj = 1
   local outi = 28
   local outj = 28
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

