luaunit = require('luaunit')

require 'nn'
require 'cltorch'
require 'clnn'

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
  N = 2
  in_size = 10
--  num_classes = 12
  
  local input = torch.Tensor(N, in_size):uniform() - 0.5
  local target = torch.multinomial(torch.range(1,in_size), N, true)

  print('input', input)
  print('target', target)
  local output = net:forward(input, target)

  print('cl')
  local netCl = net:clone():cl()
  local inputCl = input:clone():cl()
  local targetCl = target:clone():cl()
  local outputCl = netCl:forward(inputCl, targetCl)

  print('output\n', output)
  print('outputcl\n', outputCl)
--  luaunit.assertEquals(output, outputCl)
  assertFloatNear(output, outputCl)

--  local gradOutput = torch.Tensor(N, out_size):uniform() - 0.5
  local gradInput = net:backward(input, target)

  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, example_target)

  luaunit.assertEquals(gradInput, gradInputCl:double())
  collectgarbage()
end

function test_ClassNLLCriterion()
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
--  local net = nn.Sigmoid()
  local input = torch.Tensor(N, in_size):uniform() - 0.5
  local output = net:forward(input)
--  print('output\n', output)

  local netCl = net:clone():cl()
  local inputCl = input:clone():cl()
  local outputCl = netCl:forward(inputCl)
--  print('outputCl\n', outputCl)

  luaunit.assertEquals(output, outputCl:double())

  local gradOutput = torch.Tensor(N, out_size):uniform() - 0.5
  local gradInput = net:backward(input, gradOutput)
--  print('gradInput\n', gradInput)

  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, gradOutputCl)
--  print('gradInputcl\n', gradInputCl)

  luaunit.assertEquals(gradInput, gradInputCl:double())
  collectgarbage()
end

function test_linear()
  _testVectorLayer(nn.Linear(4,3))
end

function test_tanh()
  _testVectorLayer(nn.Tanh(), 4, 4)
end

function test_sigmoid()
  _testVectorLayer(nn.Sigmoid(), 4, 4)
end

function test_relu()
  _testVectorLayer(nn.ReLU(), 4, 4)
end

function test_LogSoftMax()
  _testVectorLayer(nn.LogSoftMax(), 4 , 4)
end

function _test4dLayer(net, inPlanes, inSize, outPlanes, outSize, debug)
  collectgarbage()
  print('net', net)
  local batchSize = 32
--  local numPlanes = 32
  if debug ~= nil then
    batchSize = 1
--    numPlanes = 2
  end
  local input = torch.Tensor(batchSize, inPlanes, inSize, inSize):uniform() - 0.5
  local gradOutput = torch.Tensor(batchSize, outPlanes, outSize, outSize):uniform() - 0.5

  local output = net:forward(input)

  local netCl = net:clone():cl()
  local inputCl = input:clone():cl()
  local outputCl = netCl:forward(inputCl)

  luaunit.assertEquals(output, outputCl:double())

  local gradInput = net:backward(input, gradOutput)

  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, gradOutputCl)

  luaunit.assertEquals(gradInput, gradInputCl:double())
  collectgarbage()
end

function test_SpatialMaxPooling()
  _test4dLayer(nn.SpatialMaxPooling(2,2,2,2), 32, 32, 32, 16)
  _test4dLayer(nn.SpatialMaxPooling(3,3,3,3), 32, 48, 32, 16)
end

function testSigmoidv2()
  _test4dLayer(nn.Sigmoid(), 32, 32, 32, 32)
end

function testTanhv2()
  _test4dLayer(nn.Tanh(), 32, 32, 32, 32)
end

function testFullyConnected()
  _test4dLayer(nn.FullyConnected(10), 8, 8, 10, 1)
end

--luaunit.LuaUnit.run()
os.exit( luaunit.LuaUnit.run() )

