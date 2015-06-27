luaunit = require('luaunit')

require 'nn'
require 'cltorch'
require 'clnn'
--require 'StatefulTimer'

function _testVectorLayer(its, net, in_size, out_size)
--  collectgarbage()
  N = 50
  if in_size == nil then
    in_size = net.weight:size(2)
  end
  if out_size == nil then
    out_size = net.weight:size(1)
  end
--  print('net', net)
--  local net = nn.Sigmoid()
  local input = torch.Tensor(N, in_size):uniform() - 0.5
--  local output = net:forward(input)
--  print('output\n', output)

  local netCl = net:clone():cl()
  local inputCl = input:clone():cl()
  
  local outputCl = netCl:forward(inputCl)
--  print('outputCl\n', outputCl)

--  luaunit.assertEquals(output, outputCl:double())

  local gradOutput = torch.Tensor(N, out_size):uniform() - 0.5
--  local gradInput = net:backward(input, gradOutput)
--  print('gradInput\n', gradInput)

  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, gradOutputCl)

--  timer = torch.Timer()
  s = nn.StatefulTimer()
  for it=1,its do
    netCl:forward(inputCl)
    s:state('f')
    netCl:backward(inputCl, gradOutputCl)
    s:state('b')
  end
--  time = timer:time().real / its
  print(net, 'forward time', s.times['f'] / its)
  print(net, 'backward time', s.times['b'] / its)
  print(net, 'total', (s.times['b'] + s.times['f']) / its)

--  print('gradInputcl\n', gradInputCl)

--  luaunit.assertEquals(gradInput, gradInputCl:double())
--  collectgarbage()
end

function _testperf(layer, tester, p1, p2)
  collectgarbage()
  its = 300
  timer = torch.Timer()
  for it=1,its do
    tester(layer, p1, p2)
  end
  time = timer:time().real / its
  print(layer, 'it time', time)
end

function test_LogSoftMax()
--  _testperf(nn.LogSoftMax(), _testVectorLayer, 65, 65)
  _testVectorLayer(300, nn.LogSoftMax(), 65, 65)
end

function _testTableLayer(its, net)
  collectgarbage()
  N = 50
  print('net', net)

  local netCl = net:clone():cl()

  local num_tables = 2
  local in_size = 5
  local out_size = in_size
  local t1 = torch.Tensor(N, in_size):uniform() * 2 - 1.0
  local t2 = torch.Tensor(N, in_size):uniform() * 2 - 1.0

  local input = {t1, t2}
  local inputCl = {t1:cl(), t2:cl()}

  local t1Cl = t1:clone():cl()
  local t2Cl = t2:clone():cl()
  local outputCl = netCl:forward(inputCl)

  local gradOutput = torch.Tensor(N, out_size):uniform() * 2 - 0.5
  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, gradOutputCl)

  s = nn.StatefulTimer()
  for it=1,its do
    netCl:forward(inputCl)
    s:state('f')
    netCl:backward(inputCl, gradOutputCl)
    s:state('b')
  end
  print(net, 'forward time', s.times['f'] / its)
  print(net, 'backward time', s.times['b'] / its)
  print(net, 'total', (s.times['b'] + s.times['f']) / its)
end

function testCMulTable()
  _testTableLayer(1000, nn.CMulTable())
end

function _testNarrow(its, net)
  collectgarbage()
  N = 50
  print('net', net)
  in_size = 384

  local input =torch.Tensor(N, in_size):uniform() * 2 - 1.0
  local inputCl = input:clone():cl()
  local netCl = net:clone():cl()

  local output = net:forward(input)
  local outputCl = netCl:forward(inputCl)

  local gradOutput = output:clone() * 3 + 0.1
  local gradInput = net:backward(input, gradOutput)

  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, gradOutputCl)

  s = nn.StatefulTimer()
  for it=1,its do
    netCl:forward(inputCl)
    s:state('f')
    netCl:backward(inputCl, gradOutputCl)
    s:state('b')
  end
  print(net, 'forward time', s.times['f'] / its)
  print(net, 'backward time', s.times['b'] / its)
  print(net, 'total', (s.times['b'] + s.times['f']) / its)
end

function testNarrow()
  _testNarrow(300, nn.Narrow(2, 128, 128))
end

cltorch.setTrace(1)
--luaunit.LuaUnit.run()
--test_LogSoftMax()
--testCMulTable()
testNarrow()
cltorch.setTrace(0)


