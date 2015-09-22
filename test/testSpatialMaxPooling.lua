local _test = clnn._test
local times = _test.times
local clnntest = _test.clnntest
local x_clnntest = _test.x_clnntest
local nloop = _test.nloop
local precision_forward = 1e-6
local precision_backward = 1e-6

function clnntest.SpatialMaxPooling_forward_batch()
   torch.manualSeed(123)
   local bs = 7
   local from = 37
   local to = from
   local ki = 4
   local kj = 3
   local si = 3
   local sj = 2
   local outi = 129
   local outj = 43
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local tm = {}
   local title = string.format('SpatialMaxPooling.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
      bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm
   
   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   local gconv = nn.SpatialMaxPooling(ki,kj,si,sj):cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function clnntest.SpatialMaxPooling_forward()
   -- FIXME test for different configs (and not just have non-deterministic tests :-P or
   -- incomplete tests)
   torch.manualSeed(123)
   local from = 37 -- math.random(1,64)
   local to = from
   local ki = 3 -- math.random(2,4)
   local kj = 3 -- math.random(2,4)
   local si = 2 -- math.random(1,4)
   local sj = 2 -- math.random(1,4)
   local outi = 43 -- math.random(32,256)
   local outj = 51 -- math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local tm = {}
   local title = string.format('SpatialMaxPooling.forward %dx%dx%d o %dx%d -> %dx%dx%d',
      from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm
   
   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   local gconv = nn.SpatialMaxPooling(ki,kj,si,sj):cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
--   print('gconv.indices:size()', gconv.indices:size())
--   print('sconv.indices:size()', sconv.indices:size())
   -- we have to mess around with the indices a bit, to combine the x and y indices, and
   -- then compare
   gindices = gconv.indices:float()
--   print('gindices', gindices)
--   print('sconv.indices', sconv.indices)
   gindices[1] = gindices[1]:add(-1):mul(kj) -- might be ki, not sure...
   gindices[2]:add(gindices[1])
--   print('gindices[2]', gindices[2])
   -- local error_ind = gconv.indices:float() - sconv.indices
   local error_ind = gindices[2] - sconv.indices
--   print('error_ind', error_ind)
   mytester:asserteq(error_ind:max(), 0, 'error on indices (forward) ')
end

function clnntest.SpatialMaxPooling_forward_ceil()
   torch.manualSeed(123)
   -- vgg in neural-scaling geometry:
   local from = 256
   local to = from
   local ki = 2
   local kj = 2
   local si = 2
   local sj = 2
   local outi = 2
   local outj = 2
   local ini = 4
   local inj = 3
   local ceil_mode = 1
   
   local tm = {}
   local title = string.format('SpatialMaxPooling.forward %dx%dx%d o %dx%d -> %dx%dx%d',
      from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm
   
   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj)
   if ceil_mode then sconv:ceil() end
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   local gconv = nn.SpatialMaxPooling(ki,kj,si,sj):cl()
   if ceil_mode then gconv:ceil() end
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   mytester:asserteq(groundtruth:size(1), rescl:size(1))
   mytester:asserteq(groundtruth:size(2), rescl:size(2))
   mytester:asserteq(groundtruth:size(3), rescl:size(3))

   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
--   print('gconv.indices:size()', gconv.indices:size())
--   print('sconv.indices:size()', sconv.indices:size())
   -- we have to mess around with the indices a bit, to combine the x and y indices, and
   -- then compare
   gindices = gconv.indices:float()
--   print('gindices', gindices)
--   print('sconv.indices', sconv.indices)
   gindices[1] = gindices[1]:add(-1):mul(kj) -- might be ki, not sure...
   gindices[2]:add(gindices[1])
--   print('gindices[2]', gindices[2])
   -- local error_ind = gconv.indices:float() - sconv.indices
   local error_ind = gindices[2] - sconv.indices
--   print('error_ind', error_ind)
   mytester:asserteq(error_ind:max(), 0, 'error on indices (forward) ')
end

function clnntest.SpatialMaxPooling_backward()
   -- FIXME test for different configs (and not just have non-deterministic tests :-P or
   -- incomplete tests)
   torch.manualSeed(123)
   local from = 32 -- math.random(1,64)
   local to = from
   local ki = 4 -- math.random(2,4)
   local kj = 3 -- math.random(2,4)
   local si = 3 -- math.random(1,4)
   local sj = 2 --math.random(1,4)
   local outi = 27 -- math.random(32,64)
   local outj = 31 -- math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local tm = {}
   local title = string.format('SpatialMaxPooling.backward %dx%dx%d o %dx%d -> %dx%dx%d',
      from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm
   
   local input = torch.randn(from,inj,ini)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   gradOutput = gradOutput:cl()
   local gconv = nn.SpatialMaxPooling(ki,kj,si,sj):cl()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundgrad
   
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function clnntest.SpatialMaxPooling_backward_batch()
   torch.manualSeed(123)
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   -- enforce testing non-atomic kernel (dW == kW) and (dH == kH)
   local si = ki
   local sj = kj
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local tm = {}
   local title = string.format('SpatialMaxPooling.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
      bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm
   
   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   gradOutput = gradOutput:cl()
   local gconv = nn.SpatialMaxPooling(ki,kj,si,sj):cl()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundgrad
   
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function x_clnntest.SpatialMaxPooling_backward_batch_atomic()
   torch.manualSeed(123)
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   -- enforce that kW ~= dW or kH ~= dH (which trigers the atomic kernel)
   local si = ki + ((math.random(0,1) == 1) and -math.random(1,ki-1) or math.random(1,2))
   local sj = kj + ((math.random(0,1) == 1) and -math.random(1,kj-1) or math.random(1,2))
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local tm = {}
   local title = string.format('SpatialMaxPooling.backward %dx%dx%dx%d o %dx%d (%dx%d) -> %dx%dx%dx%d',
      bs, from, inj, ini, kj, ki, si, sj, bs, to, outj, outi)
   times[title] = tm
   
   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   gradOutput = gradOutput:cl()
   local gconv = nn.SpatialMaxPooling(ki,kj,si,sj):cl()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundgrad
   
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

