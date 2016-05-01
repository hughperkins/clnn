local _test = clnn._test
local times = _test.times
local clnntest = _test.clnntest
local x_clnntest = _test.x_clnntest
local nloop = _test.nloop
local precision_forward = 1e-6
local precision_backward = 1e-6

function clnntest.SpatialUpSamplingNearest_forward_batch()
   torch.manualSeed(123)
   local bs = 7
   local from = 37
   local to = from
   local sf = 2
   local ini = 129
   local inj = 43
   local outi = sf*ini
   local outj = sf*inj

   local tm = {}
   local title = string.format('SpatialUpSamplingNearest.forward %dx%dx%dx%d o %d -> %dx%dx%dx%d',
      bs, from, inj, ini, sf, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialUpSamplingNearest(sf)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.SpatialUpSamplingNearest(sf):cl()
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

function clnntest.SpatialUpSamplingNearest_backward()
   -- FIXME test for different configs (and not just have non-deterministic tests :-P or
   -- incomplete tests)
   torch.manualSeed(123)
   local bs = 7
   local from = 37
   local to = from
   local sf = 2
   local ini = 129
   local inj = 43
   local outi = sf*ini
   local outj = sf*inj

   local tm = {}
   local title = string.format('SpatialUpSamplingNearest.backward %dx%dx%d o %d -> %dx%dx%d',
      from, inj, ini, sf, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialUpSamplingNearest(sf)
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
   local gconv = nn.SpatialUpSamplingNearest(sf):cl()
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

