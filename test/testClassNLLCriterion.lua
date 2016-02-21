local _test = clnn._test
local times = _test.times
local clnntest = _test.clnntest
local x_clnntest = _test.x_clnntest
local nloop = _test.nloop
local precision_forward = 0.01
local precision_backward = 0.01

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
end

function clnntest.ClassNLLCriterionSingleTargetScalar()
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
--   local ctarget = torch.ClTensor(1):fill(target)
   local cmod = nn.ClassNLLCriterion():cl()
   a:reset()
   local cout = cmod:forward(cinput,target)
   local cgin = cmod:backward(cinput,target)
   cltorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertlt(
      math.abs(fout-cout), precision_forward, 'error on output')
   local gerr = cgin:float() - fgin
   mytester:assertlt(gerr:abs():max(), precision_forward, 'error on gradInput')
end

function clnntest.ClassNLLCriterionSingleTarget()
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

