require 'clnn'

--function test_conv_geom(batchSize, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, inW, inH)
--  print('geometry batchSize=' .. batchSize .. ' nInputPlane=' .. nInputPlane .. ' nOutputPlane=' .. nOutputPlane ..
--        ' kW=' .. kW .. ' kH=' .. kH .. ' dW=' .. dW .. 'dH=' .. dH .. ' padW=' .. padW .. ' padH=' .. padH ..
--        ' inW=' .. inW .. ' inH=' .. inH)
function test_conv_geom(geometry)
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
--  print(geometry, geometry)
  print('geometry batchSize=' .. batchSize .. ' nInputPlane=' .. nInputPlane .. ' nOutputPlane=' .. nOutputPlane ..
        ' kW=' .. kW .. ' kH=' .. kH .. ' dW=' .. dW .. 'dH=' .. dH .. ' padW=' .. padW .. ' padH=' .. padH ..
        ' inW=' .. inW .. ' inH=' .. inH)
  local net = nn.SpatialConvolutionMM(nInputPlane,nOutputPlane,kW,kH,dW,dH,padW,padH):cl()
  --net.weight:fill(0)
  net.bias:fill(0)
  --net.weight[1][1] = 1
--  print('net.weight:size()', net.weight:size())
  local input = torch.ClTensor(batchSize,nInputPlane,inW,inH):uniform()
--  print('input[1]', input[1])
--  print('input[2]', input[2])
  local output = net:forward(input)
  net.finput:fill(-1)
  local output = net:forward(input)
  --print('output', output)

--  print('finput', net.finput)

  local netfloat = net:clone():float()
  local outfloat = netfloat:forward(input:float())
--  print('maxdiff', output:float():csub(outfloat):abs():max())
  if not (output:float():csub(outfloat):abs():max() <= 0.0001) then
    print('net.weight', net.weight)
    print('netfloat.weight', netfloat.weight)
    print('output:float()', output:float())
    print('outfloat', outfloat)
    print('diff')
    print(output:float():csub(outfloat))
  end
  assert(output:float():csub(outfloat):abs():max() <= 0.0001)
  print('all ok :-)')
end

function test_conv_all()
  test_conv_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=1, inH=1})
  test_conv_geom({batchSize=2, nInputPlane=1, nOutputPlane=1, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=1, inH=1})
--  test_conv_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=1, kH=1, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_conv_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=3, kH=1, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_conv_geom({batchSize=2, nInputPlane=1, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_conv_geom({batchSize=2, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_conv_geom({batchSize=5, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_conv_geom({batchSize=16, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_conv_geom({batchSize=17, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_conv_geom({batchSize=31, nInputPlane=2, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_conv_geom({batchSize=31, nInputPlane=16, nOutputPlane=2, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_conv_geom({batchSize=31, nInputPlane=16, nOutputPlane=32, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=5, inH=5})
--  test_conv_geom({batchSize=31, nInputPlane=16, nOutputPlane=32, kW=3, kH=3, dW=1, dH=1, padW=0, padH=0, inW=24, inH=37})
--  test_conv_geom({batchSize=31, nInputPlane=16, nOutputPlane=32, kW=3, kH=3, dW=1, dH=1, padW=1, padH=1, inW=24, inH=37})
--  test_conv_geom({batchSize=31, nInputPlane=64, nOutputPlane=128, kW=3, kH=3, dW=1, dH=1, padW=1, padH=1, inW=24, inH=37})

--  test_conv_geom(batchSize, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, inW, inH)
end

test_conv_all()

