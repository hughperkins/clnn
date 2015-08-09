require 'nn'
require 'cltorch'
require 'clnn'

local mytests = {}

function assertFloatNear(a, b)
  local diff = math.abs( a - b ) - 0.0001
  if diff >= 0 then
    print('fail different ', a, b)
    luaunit.assertTrue( diff < 0 )
  end
end

function torch.Tensor.__eq(self, b)
  diff = torch.ne(self, b)
  sum = torch.sum(diff)
  if sum == 0 then
    return true
  else
    print('Tensor')
    return false
  end
end

function torch.FloatTensor.__eq(self, b)
  diff = self - b
  diff = torch.abs(diff) - 0.0001
  diff = torch.gt(diff, 0)
  sum = torch.sum(diff)
  if sum == 0 then
    return true
  else
    print('FloatTensor')
    return false
  end
end

function torch.DoubleTensor.__eq(self, b)
--  print('DoubleTensor eq')
  diff = self - b
  diff = torch.abs(diff) - 0.0001
  diff = torch.gt(diff, 0)
  sum = torch.sum(diff)
  if sum == 0 then
    return true
  else
    print('DoubleTensor')
    print('sum', sum)
    return false
  end
end

function torch.ClTensor.__eq(self, b)
  diff = torch.ne(self, b)
  sum = torch.sum(diff)
  if sum == 0 then
    return true
  else
    return false
  end
end

function _testLabelCriterionLayer(net)
  collectgarbage()
  print('testlabelcrtierionlayer')
  N = 8
  in_size = 10
  
  local input = torch.Tensor(N, in_size):uniform() - 0.5
  local target = torch.multinomial(torch.range(1,in_size), N, true)

  local output = net:forward(input, target)

  local netCl = net:clone():cl()
  local inputCl = input:clone():cl()
  local targetCl = target:clone():cl()
  local outputCl = netCl:forward(inputCl, targetCl)

  assertFloatNear(output, outputCl)

  local gradInput = net:backward(input, target)
  local gradInputCl = netCl:backward(inputCl, targetCl)

  tester:asserteq(gradInput, gradInputCl:float())
  collectgarbage()
end

function mytests.testClassNLLCriterion()
  _testLabelCriterionLayer(nn.ClassNLLCriterion())
end

function _testVectorLayer(net, in_size, out_size)
  collectgarbage()
  N = 32
  if in_size == nil then
    in_size = net.weight:size(2)
  end
  if out_size == nil then
    out_size = net.weight:size(1)
  end
  print('net', net)
  local netCl = net:clone():cl()

--  local net = nn.Sigmoid()
  local input = torch.Tensor(N, in_size):uniform() - 0.5
  local output = net:forward(input)
--  print('output\n', output)

  local inputCl = input:clone():cl()
  local outputCl = netCl:forward(inputCl)
--  print('outputCl\n', outputCl)

  tester:asserteq(output, outputCl:float())

  local gradOutput = torch.Tensor(N, out_size):uniform() - 0.5
  local gradInput = net:backward(input, gradOutput)
--  print('gradInput\n', gradInput)

  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, gradOutputCl)
--  print('gradInputcl\n', gradInputCl)

  tester:asserteq(gradInput, gradInputCl:float())
  collectgarbage()
end

function mytests.testLinear()
  _testVectorLayer(nn.Linear(4,3))
end

function mytests.testTanh()
  _testVectorLayer(nn.Tanh(), 4, 4)
end

function mytests.testSigmoid()
  _testVectorLayer(nn.Sigmoid(), 4, 4)
end

function mytests.test_relu()
  _testVectorLayer(nn.ReLU(), 4, 4)
end

function mytests.test_LogSoftMax()
  _testVectorLayer(nn.LogSoftMax(), 4 , 4)
end

function _test4dLayer(net, inPlanes, inSize, outPlanes, outSize, debug)
  collectgarbage()
  print('net', net)
  local batchSize = 8
--  local numPlanes = 32
  if debug ~= nil then
    batchSize = 1
--    numPlanes = 2
  end
  local input = torch.Tensor(batchSize, inPlanes, inSize, inSize):uniform() - 0.5
  local gradOutput = torch.Tensor(batchSize, outPlanes, outSize, outSize):uniform() - 0.5
  local netCl = net:clone():cl()

  local output = net:forward(input)

  local inputCl = input:clone():cl()
  local outputCl = netCl:forward(inputCl)

  tester:asserteq(output, outputCl:float())

  local gradInput = net:backward(input, gradOutput)

  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, gradOutputCl)

  tester:asserteq(gradInput, gradInputCl:float())
  collectgarbage()
end

function mytests.testSpatialMaxPooling()
  _test4dLayer(nn.SpatialMaxPooling(3,3,2,2), 32, 13, 32, 6, false)  -- 13->6 is in alexnet
  _test4dLayer(nn.SpatialMaxPooling(2,2,2,2), 32, 32, 32, 16)
  _test4dLayer(nn.SpatialMaxPooling(3,3,3,3), 32, 48, 32, 16)
end

function mytests.testSigmoidv2()
  _test4dLayer(nn.Sigmoid(), 32, 32, 32, 32)
end

function mytests.testIdentity()
  _test4dLayer(nn.Identity(), 32, 32, 32, 32)
end

function mytests.testTanhv2()
  _test4dLayer(nn.Tanh(), 32, 32, 32, 32)
end

function _testTableLayer(net)
  collectgarbage()
  N = 5
  print('net', net)

  local netCl = net:clone():cl()

  local num_tables = 2
  local in_size = 5
  local out_size = in_size
  local t1 = torch.Tensor(N, in_size):uniform() * 2 - 1.0
  local t2 = torch.Tensor(N, in_size):uniform() * 2 - 1.0
--  print('t1\n', t1)
--  print('t2\n', t2)

  local input = {t1, t2}
  local inputCl = {t1:cl(), t2:cl()}

--  print('output\n', input)

  local output = net:forward(input)
--  print('output\n', output)

--  local t1Cl = t1:clone():cl()
--  local t2Cl = t2:clone():cl()
  local outputCl = netCl:forward(inputCl)
--  print('outputCl\n', outputCl)

  tester:asserteq(output, outputCl:float())

  local gradOutput = torch.Tensor(N, out_size):uniform() * 2 - 0.5
  local gradInput = net:backward(input, gradOutput)
--  print('gradInput\n', gradInput[1], gradInput[2])

  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, gradOutputCl)
--  print('gradInputcl\n', gradInputCl[1], gradInputCl[2])

  for i=1,num_tables do
    tester:asserteq(gradInput[i], gradInputCl[i]:float())
  end
  collectgarbage()
end

function mytests.testCMulTable()
  _testTableLayer(nn.CMulTable())
end

function mytests.testCAddTable()
  _testTableLayer(nn.CAddTable())
end

function _testNarrow(net)
  collectgarbage()
  N = 2
  print('net', net)
  in_size = 12

  local input = torch.Tensor(N, in_size):uniform() * 2 - 1.0
  local inputCl = input:clone():cl()
  local netCl = net:clone():cl()

  local output = net:forward(input)
--  print('output\n', output)

  local outputCl = netCl:forward(inputCl)

--  print('outputCl\n', outputCl)

  tester:asserteq(output, outputCl:float())

  -- local gradOutput = torch.Tensor(N, out_size):uniform() * 2 - 0.5
  local gradOutput = output:clone() * 3 + 0.1
  local gradInput = net:backward(input, gradOutput)

  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, gradOutputCl)

--  print('gradInput\n', gradInput)
--  print('gradInputCl\n', gradInputCl)

  tester:asserteq(gradInput, gradInputCl:float())
  collectgarbage()
end

function mytests.testNarrow()
  _testNarrow(nn.Narrow(2, 2, 5))
end

function go()
  nloop = n_loop or nloop
  local oldtype = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  -- initSeed(seed)
  tester = torch.Tester()
  local targettests = mytests
  if os.getenv('LIST') ~= nil then
    print('tests', tests)
    os.exit(0)
  end
  if os.getenv('TESTS') ~= nil then
    targettests = {}
    local filter = os.getenv('TESTS')
    for k, v in pairs(mytests) do
      if k == filter then
        targettests[k] = v
      end
    end
  end
  print('targettests', targettests)
  tester:add(targettests)
  tester:run(tests)
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

go()

