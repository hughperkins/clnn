local _test = clnn._test
local times = _test.times
local clnntest = _test.clnntest
local x_clnntest = _test.x_clnntest
local nloop = _test.nloop
local precision_forward = 1e-4
local precision_backward = 1e-3

local pointwise_transposed = clnn._testhelpers.pointwise_transposed

function clnntest.ELU_forward()
   torch.manualSeed(123)
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('ELU forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local sconv = nn.ELU()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.ELU():cl()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function clnntest.ELU_backward()
   torch.manualSeed(123)
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('ELU.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local gradOutput = torch.randn(size)
   local sconv = nn.ELU()
   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cl()
   gradOutput = gradOutput:cl()
   local gconv = sconv:clone():cl()
   gconv:forward(input)
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

clnntest.ELU_transposed = function()
   torch.manualSeed(123)
   pointwise_transposed(nn.ELU(), 'ELU', 1.5e-7)
end

