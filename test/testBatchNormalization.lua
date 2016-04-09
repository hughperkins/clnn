local _test = clnn._test
local times = _test.times
local clnntest = _test.clnntest
local x_clnntest = _test.x_clnntest
local nloop = _test.nloop
local jac = _test.jac
local sjac = _test.sjac

local precision_forward = 1e-6
local precision_backward = 1e-6

require 'nn'

local function BatchNormalization_forward(moduleName, dim, k)
   local planes = torch.random(1,k)
   local inputSize = { torch.random(2,32), planes }
   for i=1,dim do
      table.insert(inputSize, torch.random(1,k))
   end

   local tm = {}
   local title = moduleName .. '.forward ' .. table.concat(inputSize, 'x')
   times[title] = tm

   local input = torch.randn(table.unpack(inputSize))
   local sbnorm = nn[moduleName](planes)
   local groundtruth = sbnorm:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sbnorm:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gbnorm = nn[moduleName](planes):cl()
   gbnorm.weight = sbnorm.weight:cl()
   gbnorm.bias = sbnorm.bias:cl()
   local rescl = gbnorm:forward(input)

   a:reset()
   for i = 1,nloop do
      rescl = gbnorm:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward)')
   mytester:assertlt((gbnorm.running_mean:float() - sbnorm.running_mean):abs():max(),
      precision_forward, 'error on running_mean (forward)')
   mytester:assertlt((gbnorm.running_var:float() - sbnorm.running_var):abs():max(),
      precision_forward, 'error on running_var (forward)')
   mytester:assertlt((gbnorm.save_std:float() - sbnorm.save_std):abs():max(),
      precision_forward, 'error on running_var (forward)')
   print('finished forward ok')
end

local function BatchNormalization_forward_inference(moduleName, dim, k)
   k = 4
   local planes = torch.random(1,k)
   local inputSize = { torch.random(2,32), planes }
   for i=1,dim do
      table.insert(inputSize, torch.random(1,k))
   end

   local tm = {}
   local title = moduleName .. '.forward (evaluate) ' .. table.concat(inputSize, 'x')
   times[title] = tm

   local preinput = torch.randn(table.unpack(inputSize))
   local input = torch.randn(table.unpack(inputSize))
   local sbnorm = nn[moduleName](planes)
--   sbnorm.running_mean:normal(1, 2)
--   sbnorm.running_var:uniform(1e-3, 2)
   sbnorm:training()
   sbnorm:forward(preinput)
   sbnorm:evaluate()
   local groundtruth = sbnorm:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sbnorm:forward(input)
   end
   tm.cpu = a:time().real

   preinput = preinput:cl()
   input = input:cl()
   local gbnorm = nn[moduleName](planes):cl()
   gbnorm.weight = sbnorm.weight:cl()
   gbnorm.bias = sbnorm.bias:cl()
   gbnorm:training()
   gbnorm:forward(preinput)
   gbnorm:evaluate()
   local rescl = gbnorm:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gbnorm:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   print('rescl:float()', rescl:float())
   print('groundtruth', groundtruth)
   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward evaluate)')
end

local function BatchNormalization_backward(moduleName, dim, k, backwardFn)
   local planes = torch.random(1,k)
   local inputSize = { torch.random(2,32), planes }
   for i=1,dim do
      table.insert(inputSize, torch.random(1,k))
   end

   local tm = {}
   local title = moduleName .. '.backward ' .. table.concat(inputSize, 'x')
   times[title] = tm

   local input = torch.randn(table.unpack(inputSize))
   local gradOutput = torch.randn(table.unpack(inputSize))
   local sbnorm = nn[moduleName](planes)
   sbnorm:forward(input)
   sbnorm:zeroGradParameters()
   local groundgrad = backwardFn(sbnorm, input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sbnorm:zeroGradParameters()
      groundgrad = backwardFn(sbnorm, input, gradOutput)
   end
   local groundweight = sbnorm.gradWeight
   local groundbias = sbnorm.gradBias
   tm.cpu = a:time().real

   input = input:cl()
   gradOutput = gradOutput:cl()
   local gbnorm = nn[moduleName](planes):cl()
   gbnorm.weight = sbnorm.weight:cl()
   gbnorm.bias = sbnorm.bias:cl()
   gbnorm:forward(input)
   gbnorm:zeroGradParameters()
   local rescl = backwardFn(gbnorm, input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gbnorm:zeroGradParameters()
      rescl = backwardFn(gbnorm, input, gradOutput)
   end
   local weightcl = gbnorm.gradWeight
   local biascl = gbnorm.gradBias
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad
   local werror = weightcl:float() - groundweight
   local berror = biascl:float() - groundbias

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

function clnntest.BatchNormalization()
   BatchNormalization_forward('BatchNormalization', 0, 128)
   BatchNormalization_forward_inference('BatchNormalization', 0, 128)
   BatchNormalization_backward('BatchNormalization', 0, 128, function(m, input, gradOutput)
      return m:backward(input, gradOutput)
   end)
   BatchNormalization_backward('BatchNormalization', 0, 128, function(m, input, gradOutput)
      local gradInput = m:updateGradInput(input, gradOutput)
      m:accGradParameters(input, gradOutput)
      return gradInput
   end)
end

--function clnntest.SpatialBatchNormalization()
--   BatchNormalization_forward('SpatialBatchNormalization', 2, 64)
--   BatchNormalization_forward_inference('SpatialBatchNormalization', 2, 64)
--   BatchNormalization_backward('SpatialBatchNormalization', 2, 64, function(m, input, gradOutput)
--      return m:backward(input, gradOutput)
--   end)
--   BatchNormalization_backward('SpatialBatchNormalization', 2, 64, function(m, input, gradOutput)
--      local gradInput = m:updateGradInput(input, gradOutput)
--      m:accGradParameters(input, gradOutput)
--      return gradInput
--   end)
--end

