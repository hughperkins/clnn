require 'clnn'

function torch.ClTensor.str2(tensor)
  if tensor:dim() == 4 then
--    local s1 = tensor:size(1)
--    local s2 = tensor:size(2)
    local res = ''
    for s1 = 1,tensor:size(1) do
      res = res .. '(' .. s1 .. ',\n'
      for s2 = 1,tensor:size(2) do
        res = res .. '   ' .. s2 .. ',.,.)' .. '\n'
        res = res .. '     ' .. tensor[s1][s2]:__tostring():gsub('\n', '\n     '):gsub(' +$', '')
      end
    end
    return res
  else
    return tensor:__tostring__()
  end
end

function torch.FloatTensor.str2(tensor)
  if tensor:dim() == 4 then
--    local s1 = tensor:size(1)
--    local s2 = tensor:size(2)
    local res = ''
    for s1 = 1,tensor:size(1) do
      res = res .. '(' .. s1 .. ',\n'
      for s2 = 1,tensor:size(2) do
        res = res .. '   ' .. s2 .. ',.,.)' .. '\n'
        res = res .. '     ' .. tensor[s1][s2]:__tostring():gsub('\n', '\n     '):gsub(' +$', '')
      end
    end
    return res
  else
    return tensor:__tostring__()
  end
end

--function test_forward_geom(batchSize, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, inW, inH)
--  print('geometry batchSize=' .. batchSize .. ' nInputPlane=' .. nInputPlane .. ' nOutputPlane=' .. nOutputPlane ..
--        ' kW=' .. kW .. ' kH=' .. kH .. ' dW=' .. dW .. 'dH=' .. dH .. ' padW=' .. padW .. ' padH=' .. padH ..
--        ' inW=' .. inW .. ' inH=' .. inH)
function test_forward_geom(geometry)
  local batchSize = geometry.batchSize
  local nInputPlane = geometry.nInputPlane
  local nOutputPlane = geometry.nOutputPlane
  local kW = geometry.kW
  local kH = geometry.kH
  local dW = geometry.dW
  local dH = geometry.dH
  local padW = geometry.padW
  local padH = geometry.padH
  local inW = geometry.inW
  local inH = geometry.inH
  local biased = geometry.biased
--  print(geometry, geometry)
  print('geometry batchSize=' .. batchSize .. ' nInputPlane=' .. nInputPlane .. ' nOutputPlane=' .. nOutputPlane ..
        ' kW=' .. kW .. ' kH=' .. kH .. ' dW=' .. dW .. 'dH=' .. dH .. ' padW=' .. padW .. ' padH=' .. padH ..
        ' inW=' .. inW .. ' inH=' .. inH .. ' biased=' .. tostring(biased))
  local net = nn.SpatialConvolutionMM(nInputPlane,nOutputPlane,kW,kH,dW,dH,padW,padH):cl()
  --net.weight:fill(0)
  if biased == false or biased == 0 or biased == nil then
--    print('zeroing bias')
    net.bias:fill(0)
  end
  --net.weight[1][1] = 1
--  print('net.weight:size()', net.weight:size())
  local input = torch.ClTensor(batchSize,nInputPlane,inH,inW):uniform()
--  print('input:str2()\n' .. input:str2())
--  print('input[1]', input[1])
--  print('input[2]', input[2])
  local output = net:forward(input)
  net.finput:fill(-1)
  local output = net:forward(input)
  --print('output', output)

--  print('finput:str2()\n' .. net.finput:str2())

  local netfloat = net:clone():float()
  local outfloat = netfloat:forward(input:float())
--  print('maxdiff', output:float():csub(outfloat):abs():max())
  if not (output:float():csub(outfloat):abs():max() <= 0.0001) then
    print('net.weight:str2()\n' .. net.weight:str2())
    print('netfloat.weight:str2()\n' .. netfloat.weight:str2())
    print('output:float():str2()\n' .. output:float():str2())
    print('outfloat:str2()\n' .. outfloat:str2())
    print('diff')
    print(output:float():csub(outfloat))
  end
  assert(output:float():csub(outfloat):abs():max() <= 0.0001)
  print('all ok :-)')
end

function test_gradinput_geom(geometry)
  local batchSize = geometry.batchSize
  local nInputPlane = geometry.nInputPlane
  local nOutputPlane = geometry.nOutputPlane
  local kW = geometry.kW
  local kH = geometry.kH
  local dW = geometry.dW
  local dH = geometry.dH
  local padW = geometry.padW
  local padH = geometry.padH
  local inW = geometry.inW
  local inH = geometry.inH
  local biased = geometry.biased
--  print(geometry, geometry)
  print('geometry batchSize=' .. batchSize .. ' nInputPlane=' .. nInputPlane .. ' nOutputPlane=' .. nOutputPlane ..
        ' kW=' .. kW .. ' kH=' .. kH .. ' dW=' .. dW .. 'dH=' .. dH .. ' padW=' .. padW .. ' padH=' .. padH ..
        ' inW=' .. inW .. ' inH=' .. inH .. ' biased=' .. tostring(biased))
  local net = nn.SpatialConvolutionMM(nInputPlane,nOutputPlane,kW,kH,dW,dH,padW,padH):cl()
  --net.weight:fill(0)
  if biased == false or biased == 0 or biased == nil then
--    print('zeroing bias')
    net.bias:fill(0)
  end
  --net.weight[1][1] = 1
--  print('net.weight:size()', net.weight:size())
  local input = torch.ClTensor(batchSize,nInputPlane,inH,inW):uniform()
--  print('input:str2()\n' .. input:str2())
--  print('input[1]', input[1])
--  print('input[2]', input[2])
  local output = net:forward(input)
  local gradInput = net:updateGradInput(input, output)
--  net.finput:fill(-1)
--  print('finput:str2()\n' .. net.finput:str2())
  net.gradInput:fill(-1)
  local gradInput = net:updateGradInput(input, output)

--  print('finput:str2()\n' .. net.finput:str2())
  print('finput:str2()\n' .. net.finput:str2())
  print('gradInput:str2()\n' .. gradInput:str2())

  local netfloat = net:clone():float()
  local outfloat = netfloat:forward(input:float())
  local gradInputFloat = netfloat:updateGradInput(input:float(), outfloat)
  print('gradInputFloat:str2()\n' .. gradInputFloat:str2())
--  print('maxdiff', output:float():csub(outfloat):abs():max())
  if not (gradInput:float():csub(gradInputFloat):abs():max() <= 0.0001) then
--    print('net.weight:str2()\n' .. net.weight:str2())
--    print('netfloat.weight:str2()\n' .. netfloat.weight:str2())
--    print('gradInput:float():str2()\n' .. gradInput:float():str2())
--    print('gradInputFloat:str2()\n' .. gradInputFloat:str2())
--    print('diff')
--    print(gradInput:float():csub(gradInputFloat))
  end
  assert(gradInput:float():csub(gradInputFloat):abs():max() <= 0.0001)
  print('all ok :-)')
end

function test_forward_all()
  test_forward_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=1, inH=1})
  test_forward_geom({batchSize=2, nInputPlane=1, nOutputPlane=1, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=1, inH=1})
  test_forward_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=2, inH=1})
  test_forward_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
  test_forward_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=3, kH=1, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
  test_forward_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
  test_forward_geom({batchSize=2, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
  test_forward_geom({batchSize=5, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
  test_forward_geom({batchSize=16, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
  test_forward_geom({batchSize=17, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
  test_forward_geom({batchSize=31, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
  test_forward_geom({batchSize=31, nInputPlane=16, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
  test_forward_geom({batchSize=31, nInputPlane=16, nOutputPlane=32, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
  test_forward_geom({batchSize=31, nInputPlane=16, nOutputPlane=32, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=24, inH=37})
  test_forward_geom({batchSize=31, nInputPlane=16, nOutputPlane=32, kW=3, kH=3, dW=1, dH=1, padW=1, padH=1, inW=24, inH=37})
  test_forward_geom({batchSize=31, nInputPlane=64, nOutputPlane=128, kW=3, kH=3, dW=1, dH=1, padW=1, padH=1, inW=24, inH=37})

  test_forward_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=1, inH=1, biased=true})
  test_forward_geom({batchSize=2, nInputPlane=1, nOutputPlane=1, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=1, inH=1, biased=true})
  test_forward_geom({batchSize=31, nInputPlane=64, nOutputPlane=128, kW=3, kH=3, dW=1, dH=1, padW=1, padH=1, inW=24, inH=37, biased=true})
end

function test_gradinput_all()
  test_gradinput_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=1, inH=1})
  test_gradinput_geom({batchSize=2, nInputPlane=1, nOutputPlane=1, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=1, inH=1})
  test_gradinput_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=2, inH=1})
  test_gradinput_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=2, inH=3})
  test_gradinput_geom({batchSize=2, nInputPlane=1, nOutputPlane=1, kW=3, kH=1, dW=1, dH=1, padW=0, padH=0, inW=3, inH=1})
--  test_gradinput_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_gradinput_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=3, kH=1, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})

--  test_gradinput_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_gradinput_geom({batchSize=2, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})

--  test_gradinput_geom({batchSize=5, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_gradinput_geom({batchSize=16, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_gradinput_geom({batchSize=17, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_gradinput_geom({batchSize=31, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_gradinput_geom({batchSize=31, nInputPlane=16, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_gradinput_geom({batchSize=31, nInputPlane=16, nOutputPlane=32, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_gradinput_geom({batchSize=31, nInputPlane=16, nOutputPlane=32, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=24, inH=37})
--  test_gradinput_geom({batchSize=31, nInputPlane=16, nOutputPlane=32, kW=3, kH=3, dW=1, dH=1, padW=1, padH=1, inW=24, inH=37})
--  test_gradinput_geom({batchSize=31, nInputPlane=64, nOutputPlane=128, kW=3, kH=3, dW=1, dH=1, padW=1, padH=1, inW=24, inH=37})

--  test_gradinput_geom({batchSize=31, nInputPlane=64, nOutputPlane=128, kW=3, kH=3, dW=1, dH=1, padW=1, padH=1, inW=24, inH=37, biased=true})
end

--test_forward_all()
test_gradinput_all()

