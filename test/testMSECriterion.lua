local _test = clnn._test
local times = _test.times
local clnntest = _test.clnntest
local x_clnntest = _test.x_clnntest
local nloop = _test.nloop
local precision_forward = 0.01
local precision_backward = 0.01

function clnntest.mse()
   torch.manualSeed(123)
   for sizeAverage = 0, 1 do
      local size = math.random(3000,5000)
      local input = torch.randn(size,1,1)
      local target = torch.randn(size)
      local mod = nn.MSECriterion(sizeAverage == 1)
      
      local tm = {}
      local title = string.format('MSECriterion sizeAverage %d, %d ', sizeAverage, size)
      times[title] = tm
      
      local a = torch.Timer()
      local fout = mod:forward(input,target)
      local fgin = mod:backward(input,target):clone()
      tm.cpu = a:time().real
      
      local cinput = input:cl()
      local ctarget = target:cl()
      local cmod = nn.MSECriterion(sizeAverage == 1):cl()
      a:reset()
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)
      cltorch.synchronize()
      tm.gpu = a:time().real
      
      local tm2 = {}
      local title = string.format('MSECriterion2 sizeAverage %d, %d ',sizeAverage, size)
      times[title] = tm2
      tm2.cpu = tm.cpu
      local cinput2 = input:cl()
      local ctarget2 = target:cl()
      local cmod2 = nn.MSECriterion(sizeAverage == 1):cl()
      a:reset()
      local cout2 = cmod2:forward(cinput2, ctarget2)
--      local cout2 = cinput2.nn.MSECriterion_updateOutput(cmod,cinput2,ctarget2)
      local cgin2 = cmod2:updateGradInput(cinput2, ctarget2)
--      local cgin2 = cinput2.nn.MSECriterion_updateGradInput(cmod,cinput2,ctarget2)
      cltorch.synchronize()
      tm2.gpu = a:time().real
--      mytester:asserteq('torch.ClTensor', torch.type(cout2))
      mytester:asserteq('torch.ClTensor', torch.type(cgin2))
      
      mytester:assertlt(math.abs(fout-cout), precision_forward, 'error on output')
      local gerr = cgin:float() - fgin
      mytester:assertlt(gerr:abs():max(), precision_forward, 'error on gradInput')
      
      mytester:assertlt(math.abs(fout-cout2), precision_forward, 'error on output - 2')
      local gerr2 = cgin2:float() - fgin
      mytester:assertlt(gerr2:abs():max(), precision_forward, 'error on gradInput -2')
   end
end

function clnntest.mse_nosizeaverage()
   torch.manualSeed(123)
   for sizeAverage = 0, 1 do
      local size = math.random(3000,5000)
      local input = torch.randn(size,1,1)
      local target = torch.randn(size)
      local mod = nn.MSECriterion(sizeAverage == 0)
      
      local tm = {}
      local title = string.format('MSECriterion sizeAverage %d, %d ', sizeAverage, size)
      times[title] = tm
      
      local a = torch.Timer()
      local fout = mod:forward(input,target)
      local fgin = mod:backward(input,target):clone()
      tm.cpu = a:time().real
      
      local cinput = input:cl()
      local ctarget = target:cl()
      local cmod = nn.MSECriterion(sizeAverage == 0):cl()
      a:reset()
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)
      cltorch.synchronize()
      tm.gpu = a:time().real
      
      local tm2 = {}
      local title = string.format('MSECriterion2 sizeAverage %d, %d ',sizeAverage, size)
      times[title] = tm2
      tm2.cpu = tm.cpu
      local cinput2 = input:cl()
      local ctarget2 = target:cl()
      local cmod2 = nn.MSECriterion(sizeAverage == 0):cl()
      a:reset()
      local cout2 = cmod2:updateOutput(cinput2, ctarget2)
      local cgin2 = cmod2:updateGradInput(cinput2, ctarget2)
      cltorch.synchronize()
      tm2.gpu = a:time().real
--      mytester:asserteq('torch.ClTensor', torch.type(cout2))
      mytester:asserteq('torch.ClTensor', torch.type(cgin2))
      
      mytester:assertlt(math.abs(fout-cout), precision_forward, 'error on output')
      local gerr = cgin:float() - fgin
      mytester:assertlt(gerr:abs():max(), precision_forward, 'error on gradInput')
      
      mytester:assertlt(math.abs(fout-cout2), precision_forward, 'error on output - 2')
      local gerr2 = cgin2:float() - fgin
      mytester:assertlt(gerr2:abs():max(), precision_forward, 'error on gradInput -2')
   end
end

function clnntest.mse_variablebatchsize()
    torch.manualSeed(123)
    local input1 = torch.randn(10,1,1):cl()
    local input2 = torch.randn(20,1,1):cl()
    local target1 = input1:clone()
    local target2 = input2:clone()

    local mod = nn.MSECriterion():cl()

    mytester:assert(mod:forward(input1,target1), 0.0, 'error should be 0')
    mytester:assert(mod:forward(input2,target2), 0.0, 'error should be 0')
end


