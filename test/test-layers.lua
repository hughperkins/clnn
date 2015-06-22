luaunit = require('luaunit')

require 'nn'
require 'cltorch'
require 'clnn'

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

function _test_layer(net, in_size, out_size)
  N = 10
  if in_size == nil then
    in_size = net.weight:size(2)
  end
  if out_size == nil then
    out_size = net.weight:size(1)
  end
--  print('net\n', net, 'in_size', in_size, 'out_size', out_size)
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
end

function test_linear()
  _test_layer(nn.Linear(4,3))
end

function test_tanh()
  _test_layer(nn.Tanh(), 4, 4)
end

function test_sigmoid()
  _test_layer(nn.Sigmoid(), 4, 4)
end

--function test_relu()
--  _test_layer(nn.ReLU(), 4, 4)
--end

function _testLayerv2(net, inPlanes, inSize, outPlanes, outSize, debug)
  print('net', net)
  local batchSize = 32
--  local numPlanes = 32
  if debug ~= nil then
    batchSize = 2
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
end

function test_SpatialMaxPooling()
  _testLayerv2(nn.SpatialMaxPooling(2,2,2,2), 32, 32, 32, 16)
  _testLayerv2(nn.SpatialMaxPooling(3,3,3,3), 32, 48, 32, 16)
end

function testSigmoidv2()
  _testLayerv2(nn.Sigmoid(), 32, 32, 32, 32)
end

function testTanhv2()
  _testLayerv2(nn.Tanh(), 32, 32, 32, 32)
end

function testFullyConnected()
  _testLayerv2(nn.FullyConnected(10), 32, 16, 10, 1, true)
end

--luaunit.LuaUnit.run()
os.exit( luaunit.LuaUnit.run() )

