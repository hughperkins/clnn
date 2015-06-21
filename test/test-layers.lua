luaunit = require('luaunit')

require 'nn'

--print('1 torch.DoubleTensor.oldcopy', torch.DoubleTensor.cloldcopy)
--print('1 torch.DoubleTensor.oldcopy', torch.DoubleTensor.clnewcopy)

require 'cutorch'
require 'cunn'
require 'cltorch'
require 'clnn'

--print('torch.DoubleTensor.oldcopy', torch.DoubleTensor.cloldcopy)
--print('torch.DoubleTensor.oldcopy', torch.DoubleTensor.clnewcopy)
--function torch.DoubleTensor.copy(self, two)
--  print('copy 2')
--  print(torch.type(self))
--  print(torch.type(two))
--  torch.DoubleTensor.oldcopy(self, two)
--  return self
--end

print(torch.Tensor{3,5,2})
print(torch.Tensor{3,5,3}:cuda())
print(torch.Tensor{3,5,1}:cl())

function torch.Tensor.__eq(self, b)
--  print('======= eq begin ====')
--  print('self', self)
  diff = torch.ne(self, b)
--  print('diff', diff)
  sum = torch.sum(diff)
--  print('sum', sum)
  if sum == 0 then
--    print('======= eq end TRUE ====')
    return true
  else
    print('left\n', self)
    print('right\n', b)
    print('diff\n', self - b)
--    print('======= eq end FALSE ====')
    return false
  end
end

function torch.FloatTensor.__eq(self, b)
--  print('======= eq begin ====')
--  print('self', self)
  diff = self - b
--  print('diff1\n', diff)
  diff = torch.abs(diff) - 0.0001
  diff = torch.gt(diff, 0)
--  print('diff', diff)
  sum = torch.sum(diff)
--  print('sum', sum)
  if sum == 0 then
--    print('======= eq end TRUE ====')
    return true
  else
    print('left\n', self)
    print('right\n', b)
    print('diff\n', self - b)
--    print('======= eq end FALSE ====')
    return false
  end
end

function torch.DoubleTensor.__eq(self, b)
--  print('======= eq begin ====')
--  print('self', self)
  diff = self - b
--  print('diff1\n', diff)
  diff = torch.abs(diff) - 0.0001
  diff = torch.gt(diff, 0)
--  print('diff', diff)
  sum = torch.sum(diff)
--  print('sum', sum)
  if sum == 0 then
--    print('======= eq end TRUE ====')
    return true
  else
--    print('left\n', self)
--    print('right\n', b)
    print('sum', sum)
--    print('diff\n', self - b)
--    print('======= eq end FALSE ====')
    return false
  end
end

function torch.ClTensor.__eq(self, b)
  print('self', self)
  diff = torch.ne(self, b)
  print('diff', diff)
  sum = torch.sum(diff)
  print('sum', sum)
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
  print('net\n', net, 'in_size', in_size, 'out_size', out_size)
--  local net = nn.Sigmoid()
  local input = torch.Tensor(N, in_size):uniform() - 0.5
  local output = net:forward(input)
  print('output\n', output)

  local netCl = net:clone():cl()
  local inputCl = input:clone():cl()
  local outputCl = netCl:forward(inputCl)
  print('outputCl\n', outputCl)

  luaunit.assertEquals(output, outputCl:double())

  local gradOutput = torch.Tensor(N, out_size):uniform() - 0.5
  local gradInput = net:backward(input, gradOutput)
  print('gradInput\n', gradInput)

  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, gradOutputCl)
  print('gradInputcl\n', gradInputCl)

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

--local batchSize = 128
local batchSize = 2

local scenarios = {}
table.insert(scenarios, {name='simple', inplanes=1, insize=2, outplanes=1, filtersize=1})
table.insert(scenarios, {name='l1', inplanes=3, insize=128, outplanes=96, filtersize=11}) 
--table.insert(scenarios, {name='l2', inplanes=64, insize=64, outplanes=128, filtersize=9})
table.insert(scenarios, {name='l3', inplanes=128, insize=32, outplanes=128, filtersize=9})
table.insert(scenarios, {name='l4', inplanes=128, insize=16, outplanes=128, filtersize=7})
table.insert(scenarios, {name='l5', inplanes=384, insize=13, outplanes=384, filtersize=3})

function _test_conv_scenario(scenario)
  -- compare cuda and cl output, for same data
  -- we probably need to take care to ensure weights are the same, somehow
  collectgarbage()
  local input = torch.Tensor(batchSize, scenario.inplanes, scenario.insize, scenario.insize):uniform()
  local layer = nn.SpatialConvolutionMM(scenario.inplanes, scenario.outplanes,
    scenario.filtersize, scenario.filtersize)
  layer.padW = 0
  layer.padH = 0

  local layercl = layer:clone():cl()
  local layercuda = layer:clone():cuda()
  luaunit.assertEquals(layercl.weight:double(), layercuda.weight:double())
--  print('weights same')
  luaunit.assertEquals(layercl.bias:double(), layercuda.bias:double())
--  print('bias same')

--  print('calc cl out')
  local inputcl = input:cl()
--  print('input', torch.type(input))
--  print('layer', layer)
--  print('layer.output', layer.output)
  local outputcl = layercl:updateOutput(inputcl):double()
--  print('outputcl\n', outputcl)
  collectgarbage()

--  print('calc cuda out')
  local inputcuda = input:cuda()
  local outputcuda = layercuda:updateOutput(inputcuda):double()
--  print('outputcuda\n', outputcuda)
  collectgarbage()

  luaunit.assertEquals(outputcl, outputcuda)
  print('pass')
  collectgarbage()
end

function test_conv_all()
  for i, scenario in ipairs(scenarios) do
    print(i, scenario.name)
    _test_conv_scenario(scenario)
  end
end

os.exit( luaunit.LuaUnit.run() )

