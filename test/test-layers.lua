luaunit = require('luaunit')

require 'nn'
require 'clnn'

function test_linear()
  local N = 10
  local net = nn.Linear(4, 3)
  local input = torch.Tensor(N, 4):uniform()
  local output = net:forward(input)
  print('output\n', output)

  local netCl = net:clone():cl()
  local inputCl = input:clone():cl()
  local outputCl = netCl:forward(inputCl)
  print('outputCl\n', outputCl)

  local gradOutput = torch.Tensor(N, 3):uniform()
  local gradInput = net:backward(input, gradOutput)
  print('gradInput\n', gradInput)

  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, gradOutputCl)
  print('gradInputcl\n', gradInputCl)
end

function test_tanh()
  print('test_tanh()')

  local N = 10
  local net = nn.Tanh()
  local input = torch.Tensor(N, 4):uniform()
  local output = net:forward(input)
  print('output\n', output)

  local netCl = net:clone():cl()
  local inputCl = input:clone():cl()
  local outputCl = netCl:forward(inputCl)
  print('outputCl\n', outputCl)

  local gradOutput = torch.Tensor(N, 4):uniform()
  local gradInput = net:backward(input, gradOutput)
  print('gradInput\n', gradInput)

  local gradOutputCl = gradOutput:clone():cl()
  local gradInputCl = netCl:backward(inputCl, gradOutputCl)
  print('gradInputcl\n', gradInputCl)
end

os.exit( luaunit.LuaUnit.run() )

