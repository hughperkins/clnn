local _test = clnn._test
local times = _test.times
local clnntest = _test.clnntest
local x_clnntest = _test.x_clnntest
local nloop = _test.nloop
local precision_forward = 1e-6
local precision_backward = 1e-6

local function do_SpatialAveragePooling_forward(params)
   torch.manualSeed(123)
   local bs = params.bs
   local from = params.from
   local to = params.to
   local ki = params.ki
   local kj = params.kj
   local si = params.si
   local sj = params.sj
   local outi = params.outi
   local outj = params.outj
   local ceil_mode = params.ceil_mode or false

   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialAveragePooling.forward %dx%dx%d o %dx%d -> %dx%dx%d',
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = nil
   if bs ~= nil then
      input = torch.randn(bs,from,inj,ini)
   else
      input = torch.randn(from,inj,ini)
   end
   local sconv = nn.SpatialAveragePooling(ki,kj,si,sj)
   if ceil_mode then sconv:ceil() end
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.SpatialAveragePooling(ki,kj,si,sj):cl()
   if ceil_mode then gconv:ceil() end
   local res_gpu = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      res_gpu = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = res_gpu:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function clnntest.SpatialAveragePooling_forward()
   torch.manualSeed(123)
   local params = {}
   params.bs = nil  -- a nop, technically, I know...
   params.from = math.random(1,64)
   params.to = params.from
   params.ki = math.random(2,4)
   params.kj = math.random(2,4)
   params.si = math.random(1,ki)
   params.sj = math.random(1,kj)
   params.outi = math.random(32,256)
   params.outj = math.random(32,256)
   params.ceil_mode = false
   do_SpatialAveragePooling_forward(params)
end

function clnntest.SpatialAveragePooling_forward_ceil()
   torch.manualSeed(123)
   local params = {}
   params.from = math.random(1,64)
   params.to = params.from
   params.ki = math.random(2,4)
   params.kj = math.random(2,4)
   params.si = math.random(1,ki)
   params.sj = math.random(1,kj)
   params.outi = math.random(32,256)
   params.outj = math.random(32,256)
   params.ceil_mode = true
   do_SpatialAveragePooling_forward(params)
end

function clnntest.SpatialAveragePooling_forward_batch()
   torch.manualSeed(123)
   params.params = {}
   params.bs = math.random(4,10)
   params.from = math.random(1,64)
   params.to = params.from
   params.ki = math.random(2,4)
   params.kj = math.random(2,4)
   params.si = math.random(1,ki)
   params.sj = math.random(1,kj)
   params.outi = math.random(32,64)
   params.outj = math.random(32,64)
   params.ceil_mode = false
   do_SpatialAveragePooling_forward(params)
end

local function do_SpatialAveragePooling_backward(params)
   torch.manualSeed(123)
   local bs = params.bs
   local from = params.from
   local to = params.to
   local ki = params.ki
   local kj = params.kj
   local si = params.si
   local sj = params.sj
   local outi = params.outi
   local outj = params.outj
   local ceil_mode = params.ceil_mode or false

   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialMaxPooling.backward %dx%dx%d o %dx%d -> %dx%dx%d',
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = nil
   if bs ~= nil then
      input = torch.randn(bs,from,inj,ini)
   else
      input = torch.randn(from,inj,ini)
   end
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialAveragePooling(ki,kj,si,sj)
   if ceil_mode then sconv:ceil() end
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
   local gconv = nn.SpatialAveragePooling(ki,kj,si,sj):cl()
   if ceil_mode then gconv:ceil() end
   gconv:forward(input)
   gconv:zeroGradParameters()
   local res_gpu = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      res_gpu = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = res_gpu:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function clnntest.SpatialAveragePooling_backward()
   torch.manualSeed(123)
   local params = {}
   params.bs = nil  -- a nop, technically, I know...
   params.from = math.random(1,64)
   params.to = params.from
   params.ki = math.random(2,4)
   params.kj = math.random(2,4)
   params.si = math.random(1,params.ki)
   params.sj = math.random(1,params.kj)
   params.outi = math.random(32,256)
   params.outj = math.random(32,256)
   params.ceil_mode = false
   do_SpatialAveragePooling_backward(params)
end

function clnntest.SpatialAveragePooling_backward_ceil()
   torch.manualSeed(123)
   local params = {}
   params.bs = nil  -- a nop, technically, I know...
   params.from = math.random(1,64)
   params.to = params.from
   params.ki = math.random(2,4)
   params.kj = math.random(2,4)
   params.si = math.random(1,params.ki)
   params.sj = math.random(1,params.kj)
   params.outi = math.random(32,256)
   params.outj = math.random(32,256)
   params.ceil_mode = true
   do_SpatialAveragePooling_backward(params)
end

function clnntest.SpatialAveragePooling_backward_batch()
   torch.manualSeed(123)
   local params = {}
   params.bs = math.random(4,10)
   params.from = math.random(1,64)
   params.to = params.from
   params.ki = math.random(2,4)
   params.kj = math.random(2,4)
   params.si = math.random(1,params.ki)
   params.sj = math.random(1,params.kj)
   params.outi = math.random(32,64)
   params.outj = math.random(32,64)
   params.ceil_mode = false
   do_SpatialAveragePooling_backward(params)
end

function clnntest.SpatialAveragePooling_backward_batch_ceil()
   torch.manualSeed(123)
   local params = {}
   params.bs = math.random(4,10)
   params.from = math.random(1,64)
   params.to = params.from
   params.ki = math.random(2,4)
   params.kj = math.random(2,4)
   params.si = math.random(1,params.ki)
   params.sj = math.random(1,params.kj)
   params.outi = math.random(32,64)
   params.outj = math.random(32,64)
   params.ceil_mode = true
   do_SpatialAveragePooling_backward(params)
end

