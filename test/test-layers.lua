luaunit = require('luaunit')

require 'nn'
require 'cutorch'
require 'cunn'
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

local batchSize = 32
-- local batchSize = 2

local scenarios = {}
table.insert(scenarios, {name='simple', inplanes=2, insize=5, outplanes=2, filtersize=3})
table.insert(scenarios, {name='l1', inplanes=3, insize=128, outplanes=96, filtersize=11}) 
----table.insert(scenarios, {name='l2', inplanes=64, insize=64, outplanes=128, filtersize=9})
table.insert(scenarios, {name='l3', inplanes=128, insize=32, outplanes=128, filtersize=9})
table.insert(scenarios, {name='l4', inplanes=128, insize=16, outplanes=128, filtersize=7})
table.insert(scenarios, {name='l5', inplanes=384, insize=13, outplanes=384, filtersize=3})

function _test_conv_scenario(scenario)
  -- compare cuda and cl output, for same data
  collectgarbage()
  torch.manualSeed(41446)
  local layer = nn.SpatialConvolutionMM(scenario.inplanes, scenario.outplanes,
    scenario.filtersize, scenario.filtersize)
  layer.padW = 0
  layer.padH = 0

  local input = torch.Tensor(batchSize, scenario.inplanes, scenario.insize, scenario.insize):uniform():add(-0.5)

  local layercl = layer:clone():cl()
  local inputcl = input:cl()
  local outputcl = layercl:updateOutput(inputcl)
  layercl:updateGradInput(inputcl, outputcl)
  local gradInputcl = layercl.gradInput:double()
  layercl:zeroGradParameters()
  layercl:accGradParameters(inputcl, outputcl)
  local gradWeightcl = layercl.gradWeight:double()
  local gradBiascl = layercl.gradBias:double()
  local outputcl = outputcl:double()
--  layercl = nil
--  inputcl = nil
  collectgarbage()

  local layercuda = layer:clone():cuda()
  local inputcuda = input:cuda()
  local outputcuda = layercuda:updateOutput(inputcuda)
  layercuda:updateGradInput(inputcuda, outputcuda)
  local gradInputcuda = layercuda.gradInput:double()
  layercuda:zeroGradParameters()
  layercuda:accGradParameters(inputcuda, outputcuda)
  local gradWeightcuda = layercuda.gradWeight:double()
  local gradBiascuda = layercuda.gradBias:double()
  local outputcuda = outputcuda:double()
--  layercuda = nil
--  inputcuda = nil
  collectgarbage()

  luaunit.assertTrue(outputcl == outputcuda)
  luaunit.assertTrue(gradInputcl == gradInputcuda)

  weightnormalizer = 1 / torch.abs(gradWeightcuda):mean()
  gradWeightcl:mul(weightnormalizer)
  gradWeightcuda:mul(weightnormalizer)

  biasnormalizer = 1 / gradBiascuda:mean()
  gradBiascl:mul(biasnormalizer)
  gradBiascuda:mul(biasnormalizer)

  luaunit.assertTrue(gradBiascl == gradBiascuda)
  luaunit.assertTrue(gradWeightcl == gradWeightcuda)
  print('pass')
  collectgarbage()
end

function test_conv_all()
  for i, scenario in ipairs(scenarios) do
    print(i, scenario.name)
    _test_conv_scenario(scenario)
  end
end

--luaunit.LuaUnit.run()
os.exit( luaunit.LuaUnit.run() )

