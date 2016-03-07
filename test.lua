local clnntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 1
local times = {}

--e.g.: th -lclnn -e "nn.testcl{'copies'}"
--      th -lclnn -e 'clnn.tests.printExcluded()'
--      th -lclnn -e 'clnn.tests.printIncluded()'

local x_clnntest = {} -- assign to this to exclude from tests
-- I guess we can have an option to include
-- the _x assigned methods to the targets
-- just for one session

-- ======= Working tests go in this section ====================================

clnn._test = {}
clnn._test.clnntest = clnntest
clnn._test.x_clnntest = x_clnntest
clnn._test.times = times
clnn._test.nloop = nloop
clnn._test.precision_forward = precision_forward
clnn._test.precision_backward = precision_backward

-- hack tester, so it doesnt eat our assert stacktraces, where we are using a helper method
function torch.Tester:assert_sub (condition, message)
   self.countasserts = self.countasserts + 1
   if not condition then
      local ss = debug.traceback('tester',2)
      self.errors[#self.errors+1] = self.curtestname .. '\n' .. (message or '') .. '\n' .. ss .. '\n'
   end
end

include 'testClassNLLCriterion.lua'
include 'testLookupTable.lua'
include 'testSpatialAveragePooling.lua'
include 'testLogSoftMax.lua'
include 'testSoftMax.lua'
include 'testMSECriterion.lua'
include 'testSpatialMaxPooling.lua'
include 'testSpatialConvolutionMM.lua'
include 'testSpatialUpSamplingNearest.lua'

local function pointwise_transposed(proto_module, name, max_error)
   max_error = max_error or 1e-7
   local tm = {}
   local title = name .. '.transposed'
   times[title] = tm

   local input = torch.Tensor(11, 19):uniform(-1, 1)
   if name == 'Sqrt' then
      input:uniform(0.1, 1)
   end
   local inputCl = input:clone():cl()

   local cl_module = proto_module:clone():cl()

   -- transpose the inputs and DON'T make contiguous
   input = input:transpose(1, 2)
   inputCl = inputCl:transpose(1, 2)

   local output = proto_module:forward(input)
   local outputCl = cl_module:forward(inputCl)

   local error = outputCl:float() - output
   mytester:assertlt(error:abs():max(), max_error, 'error on state (forward) ')

   local gradOutput = torch.Tensor(11, 19):uniform(-1, 1)
   local gradOutputCl = gradOutput:clone():cl()

   gradOutput = gradOutput:transpose(1, 2)
   gradOutputCl = gradOutputCl:transpose(1, 2)

   local gradInput = proto_module:backward(input, gradOutput)
   local gradInputCl = cl_module:backward(inputCl, gradOutputCl)

   local error = gradInputCl:float() - gradInput
   mytester:assertlt(error:abs():max(), max_error, 'error on state (backward) ')
end

function clnntest.Tanh_forward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('Tanh forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local sconv = nn.Tanh()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.Tanh():cl()
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

function clnntest.Tanh_backward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('Tanh.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local gradOutput = torch.randn(size)
   local sconv = nn.Tanh()
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
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

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

function clnntest.Abs_forward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('Abs forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local sconv = nn.Abs()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.Abs():cl()
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

function clnntest.Abs_backward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('Abs.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local gradOutput = torch.randn(size)
   local sconv = nn.Abs()
   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cl()
   gradOutput = gradOutput:cl()
   local gconv = nn.Abs():cl()
   gconv:forward(input)
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

clnntest.Abs_transposed = function()
   pointwise_transposed(nn.Abs(), 'Abs')
end

function clnntest.Sigmoid_forward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('Sigmoid forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local sconv = nn.Sigmoid()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.Sigmoid():cl()
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

function clnntest.Sigmoid_backward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('Sigmoid.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local gradOutput = torch.randn(size)
   local sconv = nn.Sigmoid()
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
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function clnntest.LogSigmoid_forward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('LogSigmoid forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local sconv = nn.LogSigmoid()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.LogSigmoid():cl()
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

function clnntest.LogSigmoid_backward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('LogSigmoid.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local gradOutput = torch.randn(size)
   local sconv = nn.LogSigmoid()
   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cl()
   gradOutput = gradOutput:cl()
   local gconv = nn.LogSigmoid():cl()
   gconv:forward(input)
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

clnntest.LogSigmoid_transposed = function()
   pointwise_transposed(nn.LogSigmoid(), 'LogSigmoid', 1e-6)
end

local function Threshold_forward(inplace)
   inplace = inplace or false
   local size = math.random(1,100)
   local thres = torch.uniform(-1,1)
   local val = torch.uniform(-1,1)
   -- if inplace, make sure val <= thres
   if (inplace) then
      val = thres - torch.uniform(0, 1)
   end

   local tm = {}
   local title = string.format('Threshold forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local sconv = nn.Threshold(thres,val)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = sconv:cl()
   gconv.inplace = inplace
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

local function Threshold_backward(inplace)
   inplace = inplace or false
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('Threshold.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local gradOutput = torch.randn(size)
   local sconv = nn.Threshold()
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
   gconv.inplace = inplace
   gconv:forward(input)
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

clnntest.Threshold_forward = function()
   Threshold_forward()
end

clnntest.Threshold_forward_inplace = function()
   Threshold_forward(true)
end

clnntest.Threshold_backward = function()
   Threshold_backward()
end

clnntest.Threshold_backward_inplace = function()
   Threshold_backward(true)
end

clnntest.Threshold_transposed = function()
   pointwise_transposed(nn.Threshold(), "Threshold")
end

function clnntest.Sqrt_forward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('Sqrt forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size):abs()
   local sconv = nn.Sqrt()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.Sqrt():cl()
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

function clnntest.Sqrt_backward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('Sqrt.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size):abs()
   local gradOutput = torch.randn(size)
   local sconv = nn.Sqrt()
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
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function clnntest.Sqrt_zero()
   local size = math.random(1, 100)

   -- Test zero inputs; we will avoid a div-by-zero by setting to zero
   local module_gpu = nn.Sqrt():cl()
   local input_gpu = torch.ClTensor(size, size):zero()
   module_gpu:forward(input_gpu)

   local gradOutput_gpu = torch.ClTensor(size, size):fill(1)
   local gradInput_gpu = module_gpu:backward(input_gpu, gradOutput_gpu)

   mytester:assertTensorEq(gradInput_gpu:float(),
      torch.FloatTensor(size, size):zero(),
      0.000001, "error in sqrt backward singularity")

   -- Verify CPU and GPU zero behavior equivalency
   local module_cpu = nn.Sqrt()
   local input_cpu = input_gpu:float()
   module_cpu:forward(input_cpu)

   local gradOutput_cpu = gradOutput_gpu:float()
   local gradInput_cpu = module_cpu:backward(input_cpu, gradOutput_cpu)

   mytester:assertTensorEq(gradInput_gpu:float(),
      gradInput_cpu:float(),
      0.000001, "Sqrt_zero CPU and GPU not equivalent")
end

function clnntest.Square_forward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('Square forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local sconv = nn.Square()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.Square():cl()
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

function clnntest.Square_backward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('Square.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local gradOutput = torch.randn(size)
   local sconv = nn.Square()
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
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

clnntest.Square_transposed = function()
   pointwise_transposed(nn.Square(), 'Square')
end

function clnntest.Sum_forward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Sum forward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local sconv = nn.Sum(2)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.Sum(2):cl()
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

function clnntest.Sum_backward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Sum.backward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local gradOutput = torch.randn(size1)
   local sconv = nn.Sum(2)
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
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad
<<<<<<< HEAD

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
=======

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function clnntest.ClassNLLCriterionMultipleTarget()
   local size = 3000
   local input = torch.randn(size, size)
   local target = torch.randperm(size)
   local mod = nn.ClassNLLCriterion()

   local tm = {}
   local title = string.format('ClassNLLCriterionMultiTarget %d ',size)
   times[title] = tm

   local a = torch.Timer()
   local fout = mod:forward(input, target)
   local fgin = mod:backward(input, target):clone()
   tm.cpu = a:time().real

   local cinput = input:cl()
   local ctarget = target:cl()
   local cmod = nn.ClassNLLCriterion():cl()
   a:reset()
   local cout = cmod:forward(cinput,ctarget)
   local cgin = cmod:backward(cinput,ctarget)
   cltorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertlt(
      math.abs(fout-cout), precision_forward, 'error on output')

   local gerr = cgin:float() - fgin
   mytester:assertlt(gerr:abs():max(), precision_forward * 3, 'error on gradInput')
>>>>>>> 64ebe3541136d24babb387142c09a0424361f162
end

function clnntest.CMul_forward_batch()
   local bs = math.random(8,32)
   local nini = math.random(1,100)
   local ninj = math.random(1,100)
   local nink = math.random(1,100)

   local tm = {}
   local title = string.format('CMul forward %d %d %d %d', bs, nini, ninj, nink)
   times[title] = tm

   local input = torch.randn(bs, nini, ninj, nink)
   local sconv = nn.CMul(nini, ninj, nink)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = sconv:clone():cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) batch ')
end

clnntest.Tanh_transposed = function()
   pointwise_transposed(nn.Tanh(), 'Tanh', 1.5e-7)
end

clnntest.Sigmoid_transposed = function()
   pointwise_transposed(nn.Sigmoid(), 'Sigmoid', 1.5e-07)
end

clnntest.Sqrt_transposed = function()
   pointwise_transposed(nn.Sqrt(), 'Sqrt', 1.5e-07)
end

-- ======= Failing tests go in this section ====================================

function x_clnntest.copies()
   -- test vector
   local t = torch.ClTensor(100,10)

   -- simple copy
   t:normal()
   local t2 = t:clone()
   mytester:asserteq( t:add(-1,t2):abs():max(), 0, 'simple copy')

   -- transpose copy
   t:normal()
   local t3 = t:transpose(1,2)
   local t4 = t3:clone()
   mytester:asserteq( t3:add(-1,t4):abs():max(), 0, 'transpose copy')

   -- unfold copy
   t:normal()
   local t5 = t:unfold(2,5,1)
   local t6 = t5:clone()
   mytester:asserteq( t5:add(-1,t6):abs():max(), 0, 'transpose copy')

   -- host copy
   t = torch.FloatTensor(100,10)
   t:normal()
   local tc = t:cl()
   tc = tc:transpose(1,2)
   local t2 = tc:float()
   mytester:asserteq(t:transpose(1,2):add(-1,t2):abs():max(), 0, 'host copy, plus transpoe')
end

--[[
function clnntest.Euclidean_forward_batch()
   local bs = math.random(8,32)
   local nin = math.random(1,100)
   local nout = math.random(1,100)

   local tm = {}
   local title = string.format('Euclidean forward %d %d -> %d %d', bs, nin, bs, nout)
   times[title] = tm

   local input = torch.randn(bs, nin)
   local sconv = nn.Euclidean(nin, nout)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = sconv:clone():cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) batch ')
end

function clnntest.Euclidean_backward_batch()
   local bs = math.random(8,32)
   local nin = math.random(1,100)
   local nout = math.random(1,100)

   local tm = {}
   local title = string.format('Euclidean backward %d %d <- %d %d', bs, nin, bs, nout)
   times[title] = tm

   local input = torch.randn(bs, nin)
   local gradOutput = torch.randn(bs, nout)
   local sconv = nn.Euclidean(nin, nout)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   tm.cpu = a:time().real

   input = input:cl()
   gradOutput = gradOutput:cl()
   local gconv = sconv:clone():cl()
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

   local weightcl = gconv.gradWeight

   local error = rescl:float() - groundgrad
   local werror = weightcl:float() - groundweight

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
end
]]--

--[[
function clnntest.WeightedEuclidean_forward_batch()
   local bs = math.random(8,32)
   local nin = math.random(1,100)
   local nout = math.random(1,100)

   local tm = {}
   local title = string.format('WeightedEuclidean forward %d %d -> %d %d', bs, nin, bs, nout)
   times[title] = tm

   local input = torch.randn(bs, nin)
   local sconv = nn.WeightedEuclidean(nin, nout)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = sconv:clone():cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) batch ')
end

function clnntest.WeightedEuclidean_backward_batch()
   local bs = math.random(8,32)
   local nin = math.random(1,100)
   local nout = math.random(1,100)

   local tm = {}
   local title = string.format('WeightedEuclidean backward %d %d <- %d %d', bs, nin, bs, nout)
   times[title] = tm

   local input = torch.randn(bs, nin)
   local gradOutput = torch.randn(bs, nout)
   local sconv = nn.WeightedEuclidean(nin, nout)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local grounddiagCov = sconv.gradDiagCov
   tm.cpu = a:time().real

   input = input:cl()
   gradOutput = gradOutput:cl()
   local gconv = sconv:clone():cl()
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

   local weightcl = gconv.gradWeight
   local diagCovcl = gconv.gradDiagCov

   local error = rescl:float() - groundgrad
   local werror = weightcl:float() - groundweight
   local derror = diagCovcl:float() - grounddiagCov

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(derror:abs():max(), precision_backward, 'error on diagCov (backward) ')
end
]]--

function x_clnntest.Max_forward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Max forward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local sconv = nn.Max(2)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.Max(2):cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')

   local error = gconv.indices:float() - sconv.indices
   mytester:assertlt(error:abs():max(), 1e-8, 'error on indices ')
end

function x_clnntest.Max_backward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Max.backward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local gradOutput = torch.randn(size1)
   local sconv = nn.Max(2)
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
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function x_clnntest.Min_forward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Min forward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local sconv = nn.Min(2)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.Min(2):cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')

   local error = gconv.indices:float() - sconv.indices
   mytester:assertlt(error:abs():max(), 1e-8, 'error on indices ')
end

function x_clnntest.Min_backward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Min.backward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local gradOutput = torch.randn(size1)
   local sconv = nn.Min(2)
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
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

--[[
function clnntest.Mean_forward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Mean forward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local sconv = nn.Mean(2)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.Mean(2):cl()
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

function clnntest.Mean_backward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Mean.backward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local gradOutput = torch.randn(size1)
   local sconv = nn.Mean(2)
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
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end
]]--

function x_clnntest.SpatialSubSampling_forward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialSubSampling.forward %dx%dx%d o %dx%d -> %dx%dx%d',
      from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj):cl()
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
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function x_clnntest.SpatialSubSampling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialSubSampling.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
      bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj):cl()
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
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function x_clnntest.SpatialSubSampling_backward()
   local from = 37 -- math.random(1,64)
   local to = from
   -- FIXME test for different configs (and not just have non-deterministic tests :-P or
   -- incomplete tests)
   local ki = 4 -- math.random(2,4)
   local kj = 3 -- math.random(2,4)
   local si = 3 -- math.random(2,4)
   local sj = 4 -- math.random(2,4)
   local outi = 41 -- math.random(32,64)
   local outj = 52 -- math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialSubSampling.backward %dx%dx%d o %dx%d -> %dx%dx%d',
      from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj)
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
   local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj):cl()
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

function x_clnntest.SpatialSubSampling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialSubSampling.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
      bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
   local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj)
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
   local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj):cl()
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

function x_clnntest.SpatialAdaptiveMaxPooling_forward()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   local tm = {}
   local title = string.format('SpatialAdaptiveMaxPooling.forward %dx%dx%d -> %dx%dx%d',
      from, inj, ini, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   local error_ind = gconv.indices:float() - sconv.indices
   mytester:asserteq(error_ind:max(), 0, 'error on indices (forward) ')
end

function x_clnntest.SpatialAdaptiveMaxPooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   local tm = {}
   local title = string.format('SpatialAdaptiveMaxPooling.forward %dx%dx%dx%d -> %dx%dx%dx%d',
      bs, from, inj, ini, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):cl()
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

function x_clnntest.SpatialAdaptiveMaxPooling_backward()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   local tm = {}
   local title = string.format('SpatialAdaptiveMaxPooling.backward %dx%dx%d -> %dx%dx%d',
      from, inj, ini, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj)
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
   local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):cl()
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

function x_clnntest.SpatialAdaptiveMaxPooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   local tm = {}
   local title = string.format('SpatialAdaptiveMaxPooling.backward %dx%dx%dx%d -> %dx%dx%dx%d',
      bs, from, inj, ini, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
   local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj)
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
   local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):cl()
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

function x_clnntest.SpatialLPPooling_forward()
   local from = math.random(1,64)
   local to = from
   local pnorm = 2
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialLPPooling.forward (P=2 only) %dx%dx%d o %dx%d -> %dx%dx%d',
      from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialLPPooling(from,pnorm,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.SpatialLPPooling(from,pnorm,ki,kj,si,sj):cl()
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

function x_clnntest.SpatialLPPooling_backward()
   local from = math.random(1,64)
   local to = from
   local pnorm = 2
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialLPPooling.backward (P=2 only) %dx%dx%d o %dx%d -> %dx%dx%d',
      from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialLPPooling(from,pnorm,ki,kj,si,sj)
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
   local gconv = sconv:clone():cl()
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

function x_clnntest.distkldiv()
   for sizeAverage = 0, 1 do
      local size = math.random(3000,5000)
      local input = torch.randn(size,1,1)
      local target = torch.randn(size)
      local mod = nn.DistKLDivCriterion(sizeAverage == 1)

      local tm = {}
      local title = string.format('DistKLDivCriterion sizeAverage %d, %d ',sizeAverage,size)
      times[title] = tm

      local a = torch.Timer()
      local fout = mod:forward(input,target)
      local fgin = mod:backward(input,target):clone()
      tm.cpu = a:time().real

      local cinput = input:cl()
      local ctarget = target:cl()
      local cmod = nn.DistKLDivCriterion(sizeAverage == 1):cl()
      a:reset()
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)
      cltorch.synchronize()
      tm.gpu = a:time().real

      mytester:assertlt(math.abs(fout-cout), precision_forward, 'error on output')
      local gerr = cgin:float() - fgin
      mytester:assertlt(gerr:abs():max(), precision_backward, 'error on gradInput')
   end
end

function x_clnntest.TemporalConvolution_forward()
   local from = math.random(1,64) -- inputFrameSize
   local to = math.random(1,64) -- outputFrameSize
   local ki = math.random(3,15) -- kernelWidth (kW)
   local si = math.random(1,2) -- stepSize (dW)
   local outi = math.random(1,256) -- nOutputFrame
   local ini = (outi-1)*si+ki -- nInputFrame

   local tm = {}
   local title = string.format('TemporalConvolution.forward %dx%d o %d -> %dx%d [s: %d]',
      from, ini, ki, to, outi, si)
   times[title] = tm

   local input = torch.randn(ini,from)
   local sconv = nn.TemporalConvolution(from,to,ki,si)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.TemporalConvolution(from,to,ki,si):cl()
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
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function x_clnntest.TemporalConvolution_forward_batch()
   local bs = math.random(4,16)
   local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local si = math.random(1,2)
   local outi = math.random(1,256)
   local ini = (outi-1)*si+ki

   local tm = {}
   local title = string.format('TemporalConvolution.forward %dx%dx%d o %d -> %dx%dx%d [s: %d]',
      bs, from, ini, ki, bs, to, outi, si)
   times[title] = tm

   local input = torch.randn(bs,ini,from)
   local sconv = nn.TemporalConvolution(from,to,ki,si)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.TemporalConvolution(from,to,ki,si):cl()
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
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function x_clnntest.TemporalConvolution_backward()
   local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local si = math.random(1,2)
   local outi = math.random(1,256)
   local ini = (outi-1)*si+ki

   local tm = {}
   local title = string.format('TemporalConvolution.backward %dx%d o %d -> %dx%d',
      from, ini, ki, to, outi)

   times[title] = tm

   local input = torch.randn(ini,from)
   local gradOutput = torch.randn(outi,to)
   local sconv = nn.TemporalConvolution(from,to,ki,si)
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
   local gconv = nn.TemporalConvolution(from,to,ki,si):cl()
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

function x_clnntest.TemporalConvolution_backward_batch()
   local bs = math.random(4,16)
   local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local si = math.random(1,2)
   local outi = math.random(1,256)
   local ini = (outi-1)*si+ki

   local tm = {}
   local title = string.format('TemporalConvolution.backward %dx%dx%d o %d -> %dx%dx%d',
      bs, from, ini, ki, bs, to, outi)
   times[title] = tm

   local input = torch.randn(bs,ini,from)
   local gradOutput = torch.randn(bs,outi,to)
   local sconv = nn.TemporalConvolution(from,to,ki,si)
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
   local gconv = nn.TemporalConvolution(from,to,ki,si):cl()
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

--[[
function clnntest.Dropout()
   local p = 0.2 --prob of droping out a neuron
   local input = torch.ClTensor(1000):fill((1-p))
   local module = nn.Dropout(p)
   module:cl()
   -- version 2
   local output = module:forward(input)
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input, input)
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')
   -- version 1 (old nnx version)
   local input = input:fill(1)
   local module = nn.Dropout(p,true)
   module:cl()
   local output = module:forward(input)
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input, input)
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')
end

function clnntest.Dropout_forward()
   local size = math.random(1,200)

   local tm = {}
   local title = string.format('Dropout forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local sconv = nn.Dropout()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.Dropout():cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

end
]]--

function x_clnntest.SoftPlus_forward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('SoftPlus forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local sconv = nn.SoftPlus()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.SoftPlus():cl()
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

function x_clnntest.SoftPlus_backward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('SoftPlus.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local gradOutput = torch.randn(size)
   local sconv = nn.SoftPlus()
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
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function x_clnntest.SpatialUpSamplingNearest_forward()
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   local tm = {}
   local title = string.format('SpatialUpSamplingNearest.forward %dx%dx%d -> %dx%dx%d',
      f, h, w, f, h*scale, w*scale)
   times[title] = tm

   local input = torch.randn(f, h, w)
   local sconv = nn.SpatialUpSamplingNearest(scale)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = sconv:clone():cl()
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

function x_clnntest.SpatialUpSamplingNearest_forward_batch()
   local nbatch = torch.random(3, 15)
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   local tm = {}
   local title = string.format('SpatialUpSamplingNearest.forward %dx%dx%dx%d -> %dx%dx%dx%d',
      nbatch, f, h, w, nbatch, f, h*scale, w*scale)
   times[title] = tm

   local input = torch.randn(nbatch, f, h, w)
   local sconv = nn.SpatialUpSamplingNearest(scale)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = sconv:clone():cl()
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

function x_clnntest.SpatialUpSamplingNearest_backward()
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   local tm = {}
   local title = string.format('SpatialUpSamplingNearest.backward %dx%dx%d -> %dx%dx%d',
      f, h, w, f, h*scale, w*scale)
   times[title] = tm

   local input = torch.randn(f, h, w)
   local gradOutput = torch.randn(f, h*scale, w*scale)
   local sconv = nn.SpatialUpSamplingNearest(scale)
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
   local gconv = sconv:clone():cl()
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

function x_clnntest.SpatialUpSamplingNearest_backward_batch()
   local nbatch = torch.random(3, 15)
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   local tm = {}
   local title = string.format('SpatialUpSamplingNearest.backward %dx%dx%dx%d -> %dx%dx%dx%d',
      nbatch, f, h, w, nbatch, f, h*scale, w*scale)
   times[title] = tm

   local input = torch.randn(nbatch, f, h, w)
   local gradOutput = torch.randn(nbatch, f, h*scale, w*scale)
   local sconv = nn.SpatialUpSamplingNearest(scale)
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
   local gconv = sconv:clone():cl()
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

function x_clnntest.l1cost()
   local size = math.random(300,500)
   local input = torch.randn(size)
   local mod = nn.L1Cost()

   local tm = {}
   local title = string.format('L1Cost %d ',size)
   times[title] = tm

   local a = torch.Timer()
   local fout = mod:forward(input)
   local fgin = mod:backward(input):clone()
   tm.cpu = a:time().real

   local cinput = input:cl()
   local cmod = nn.L1Cost():cl()
   a:reset()
   local cout = cmod:forward(cinput)
   local cgin = cmod:backward(cinput)
   cltorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertlt(math.abs(fout-cout), precision_forward, 'error on output')
   local gerr = cgin:float() - fgin
   mytester:assertlt(gerr:abs():max(), precision_forward, 'error on gradInput')
end


<<<<<<< HEAD
=======
function x_clnntest.ClassNLLCriterionSingleTarget()
   local size = math.random(3000,5000)
   local input = torch.randn(size)
   local target = 1
   local mod = nn.ClassNLLCriterion()

   local tm = {}
   local title = string.format('ClassNLLCriterionSingleTarget %d ',size)
   times[title] = tm

   local a = torch.Timer()
   local fout = mod:forward(input, target)
   local fgin = mod:backward(input, target):clone()
   tm.cpu = a:time().real

   local cinput = input:cl()
   local ctarget = torch.ClTensor(1):fill(target)
   local cmod = nn.ClassNLLCriterion():cl()
   a:reset()
   local cout = cmod:forward(cinput,ctarget)
   local cgin = cmod:backward(cinput,ctarget)
   cltorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertlt(
      math.abs(fout-cout), precision_forward, 'error on output')
   local gerr = cgin:float() - fgin
   mytester:assertlt(gerr:abs():max(), precision_forward, 'error on gradInput')
end

>>>>>>> 64ebe3541136d24babb387142c09a0424361f162
function x_clnntest.TemporalMaxPooling()
   local input = torch.rand(16, 18, 3)
   local settings = {{2, 2}, {3, 3}, {4, 2}, {2, 4}, {3, 5}}

   for i, setting in ipairs(settings) do
      local mod = nn.TemporalMaxPooling(setting[1], setting[2])

      local tm = {}
      local title = 'TemporalMaxPooling '..setting[1]..' '..setting[2]
      times[title] = tm

      local a = torch.Timer()
      local fout = mod:forward(input)
      local fgout = torch.rand(fout:size())
      local fgin = mod:backward(input, fgout):clone()
      tm.cpu = a:time().real

      local cinput = input:cl()
      local cgout = fgout:cl()
      local cmod = nn.TemporalMaxPooling(setting[1], setting[2]):cl()
      a:reset()
      local cout = cmod:forward(cinput)
      local cgin = cmod:backward(cinput, cgout)
      cltorch.synchronize()
      tm.gpu = a:time().real

      local outerror = cout:float() - fout
      mytester:assertlt(outerror:abs():max(), precision_forward, 'error on output')

      local ginerror = cgin:float() - fgin
      mytester:assertlt(ginerror:abs():max(), precision_backward, 'error on gradInput')
   end
end

function x_clnntest.VolumetricConvolution_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local kk = math.random(3,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local sk = math.random(1,kk)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local outk = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   local tm = {}
   local title = string.format('VolumetricConvolution.forward %dx%dx%dx%d o %dx%dx%d -> %dx%dx%dx%d',
      from, ink, inj, ini, kk, kj, ki, to, outk, outj, outi)
   times[title] = tm

   local input = torch.randn(from,ini,inj,ink)
   local sconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj):cl()
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
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function x_clnntest.VolumetricConvolution_forward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,8)
   local to = math.random(1,4) * 4
   local ki = math.random(3,8)
   local kj = math.random(3,8)
   local kk = math.random(3,8)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local sk = math.random(1,kk)
   local outi = math.random(1,16)
   local outj = math.random(1,16)
   local outk = math.random(1,16)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   local tm = {}
   local title = string.format('VolumetricConvolution.forward %dx%dx%dx%dx%d o %dx%dx%d -> %dx%dx%dx%dx%d',
      bs, from, ink, inj, ini, kk, kj, ki, bs, to, outk, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,ini,inj, ink)
   local sconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sj,sk)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input, sconv)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sj,sk):cl()
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
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function x_clnntest.VolumetricConvolution_backward_single()
   local from = math.random(1,4)
   local to = math.random(1,3) * 8
   local ki = math.random(3,8)
   local kj = math.random(3,8)
   local kk = math.random(3,8)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local sk = math.random(1,kk)
   local outi = math.random(1,16)
   local outj = math.random(1,16)
   local outk = math.random(1,16)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   local tm = {}
   local title = string.format('VolumetricConvolution.backward %dx%dx%dx%d o %dx%dx%d -> %dx%dx%dx%d',
      from, ink, inj, ini, kk, kj, ki, to, outk, outj, outi)
   times[title] = tm

   local input = torch.randn(from, ini, inj, ink)
   local gradOutput = torch.randn(to, outi, outj, outk)
   local sconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj)
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
   local gconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj):cl()
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

function x_clnntest.VolumetricConvolution_backward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,4)
   local to = math.random(1,3) * 8
   local ki = math.random(3,8)
   local kj = math.random(3,8)
   local kk = math.random(3,8)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local sk = math.random(1,kk)
   local outi = math.random(1,16)
   local outj = math.random(1,16)
   local outk = math.random(1,16)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   local tm = {}
   local title = string.format('VolumetricConvolution.backward %dx%dx%dx%dx%d o %dx%dx%d -> %dx%dx%dx%dx%d',
      bs, from, ink, inj, ini, kk, kj, ki, bs, to, outk, outj, outi)
   times[title] = tm

   local input = torch.randn(bs, from, ini, inj, ink)
   local gradOutput = torch.randn(bs, to, outi, outj, outk)
   local sconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj)
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
   local gconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj):cl()
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

function x_clnntest.CMul_backward_batch()
   local bs = math.random(8,32)
   local nini = math.random(1,100)
   local ninj = math.random(1,100)
   local nink = math.random(1,100)

   local tm = {}
   local title = string.format('CMul backward %d %d %d %d', bs, nini, ninj, nink)
   times[title] = tm

   local input = torch.randn(bs, nini, ninj, nink)
   local gradOutput = torch.randn(bs, nini, ninj, nink)
   local sconv = nn.CMul(nini, ninj, nink)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   tm.cpu = a:time().real

   input = input:cl()
   gradOutput = gradOutput:cl()
   local gconv = sconv:clone():cl()
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

   local weightcl = gconv.gradWeight

   local error = rescl:float() - groundgrad
   local werror = weightcl:float() - groundweight

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
end

function x_clnntest.PReLU_forward()
   local nOutputPlane = 8
   local w = math.random(1,100)
   local h = math.random(1,100)

   local tm = {}
   local title = string.format('PReLU forward %d x %d', w, h)
   times[title] = tm

   local input = torch.randn(nOutputPlane,h,w)
   local sconv = nn.PReLU(nOutputPlane)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = sconv:cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state')
end

function x_clnntest.PReLU_backward()
   local nOutputPlane = 8
   local w = math.random(1,10)
   local h = math.random(1,10)

   local tm = {}
   local title = string.format('PReLU backward %d x %d', w, h)
   times[title] = tm

   local input = torch.randn(nOutputPlane, h, w)
   local gradOutput = torch.randn(#input)
   local sconv = nn.PReLU(nOutputPlane)
   local gconv = sconv:clone():cl()

   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cl()
   gradOutput = gradOutput:cl()
   gconv:forward(input)
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local err = rescl:float() - groundgrad
   local weightGradError = gconv.gradWeight:float() - sconv.gradWeight

   mytester:assertlt(err:abs():max(), precision_backward, 'error on state')
   mytester:assertlt(weightGradError:abs():max(), precision_backward, 'error on weight')
end

local function setUp()
   -- cltorch.setDevice(1)
   initSeed(123456, false)
end

for k,v in pairs(clnntest) do
   clnntest[k] = function()
      setUp()
      v()
   end
end

function initSeed(seed, echo)
   if echo == nil then
      echo = true
   end
   seed = seed or os.time()
   -- ensure that you can reproduce a failing test
   if echo then
      print('seed: ', seed)
   end
   math.randomseed(seed)
   torch.manualSeed(seed)
   --cltorch.manualSeedAll(seed)
end

function nn.testcl(tests, print_timing, n_loop, seed)
   nloop = n_loop or nloop
   local oldtype = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.FloatTensor')
   -- initSeed(seed)
   mytester = torch.Tester()
   local mytests = clnntest
   if os.getenv('TESTS') ~= nil then
      mytests = {}
      for name, test in pairs(clnntest) do
         if name == os.getenv('TESTS') then
            table.insert(mytests, test)
         end
      end
   elseif os.getenv('LIST') ~= nil then
      for name, test in pairs(clnntest) do
         print(name)
      end
      os.exit(0)
   end
   mytester:add(mytests)
   mytester:run(tests)
   torch.setdefaulttensortype(oldtype)
   if print_timing then
      print ''
      print ' ------------------------------------------------------------------------------------------------'
      print '|  Module                                                                          |  Speedup    |'
      print ' ------------------------------------------------------------------------------------------------'
      for module,tm in pairs(times) do
         local str = string.format('| %-80s | %4.2f        |', module, (tm.cpu / (tm.gpu or 1e6)))
         print(str)
      end
      print ' ------------------------------------------------------------------------------------------------'
   end
end

clnn.test = nn.testcl

clnn.tests = {}
function clnn.tests.printExcluded()
   print('Excluded tests:')
   for k,v in pairs(x_clnntest) do
      print('  ' .. k)
   end
end
function clnn.tests.printIncluded()
   print('Included tests:')
   for k,v in pairs(clnntest) do
      print('  ' .. k)
   end
end
