luaunit = require('luaunit')

require 'nn'
require 'clnn'

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
    print('left\n', self)
    print('right\n', b)
    print('diff\n', self - b)
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

os.exit( luaunit.LuaUnit.run() )

