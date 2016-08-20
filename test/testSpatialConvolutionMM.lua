local _test = clnn._test
local times = _test.times
local clnntest = _test.clnntest
local x_clnntest = _test.x_clnntest
local nloop = _test.nloop
local precision_forward = 1e-4
local precision_backward = 1e-2

function clnntest.SpatialConvolutionMM_forward_1d_byhand()
  -- do a 1d convolution, as per clnntest.TemporalConvolution2_forward(), but directly on
  -- SpatialConvolutionMM

  local batchSize = 1
  local inFeatures = 1
  local outFeatures = 7
  local sentenceLength = 3
  local kernelSize = 3
  local net = nn.SpatialConvolutionMM(inFeatures, outFeatures, 1, kernelSize)
  net:cl()
  local weights = net.weight
  weights:uniform(-1, 1)
  net.bias:zero()  -- to simplify test
  local input = torch.FloatTensor(batchSize, inFeatures, sentenceLength, 1):uniform()
  input = input:cl()
  local output = net:forward(input)
--  print('weights:size()', weights:size())
  weights = weights:view(torch.LongStorage({outFeatures, inFeatures, kernelSize}))
--  print('weights:size()', weights:size())
--  print('output:size()', output:size())
  local outLength = sentenceLength - math.floor(kernelSize / 2) * 2
  local ourOut = torch.FloatTensor(batchSize, outFeatures, outLength, 1):zero()

  for b=1,batchSize do
    -- each output feature is independnet from other outputs
    for outFeature=1,outFeatures do
      -- each output point along outS dimensino is indepdnent from other outputs
      for outS=1,outLength do
        local sum = 0
        -- convolve is sum over kernel size, and over the input features
        for k=1,kernelSize do
          local inS = outS + (k - 1)
          for inFeature=1,inFeatures do
            local weight = weights[outFeature][inFeature][k]
            sum = sum + weight * input[b][inFeature][inS][1]
          end
        end
        ourOut[b][outFeature][outS][1] = sum
      end
    end
  end
--  print('output[1]')
--  print(output[1])
--  print('ourOut[1]')
--  print(ourOut[1])
--  print('output[1] - ourOut[1]')
--  print(output[1]:float() - ourOut[1])
  mytester:assertlt((output:float() - ourOut):abs():max(), 0.0001)
end

function clnntest.SpatialConvolutionMM_forward_single()
   torch.manualSeed(123)
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = 1 -- not supported by CPU version yet
   local sj = si
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local tm = {}
   local title = string.format('SpatialConvolutionMM.forward %dx%dx%d o %dx%d -> %dx%dx%d [s: %dx%d]',
      from, inj, ini, kj, ki, to, outj, outi, sj, si)
   times[title] = tm
   
   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cl()
   gconv.weight = sconv.weight:cl()
   gconv.bias = sconv.bias:cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), 1e-5, 'error on state (forward) ')
end

function clnntest.SpatialConvolutionMM_forward_single_vgglayer13()
   torch.manualSeed(123)
   local from = 256
   local to = 256
   local ki = 3
   local kj = 3
   local si = 1
   local sj = 1
   local outi = 8
   local outj = 8
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local tm = {}
   local title = string.format('SpatialConvolutionMM.forward %dx%dx%d o %dx%d -> %dx%dx%d [s: %dx%d]',
      from, inj, ini, kj, ki, to, outj, outi, sj, si)
   times[title] = tm
   
   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cl()
   gconv.weight = sconv.weight:cl()
   gconv.bias = sconv.bias:cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), 1e-5, 'error on state (forward) ')
end

function clnntest.SpatialConvolutionMM_forward_single_padded()
   torch.manualSeed(123)
   local from = 27
   local to = 5
   local ki = 3 -- VGG
   local kj = 3
   local si = 1
   local sj = si
   local outi = 31
   local outj = 25
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local tm = {}
   local title = string.format('SpatialConvolutionMM.forward %dx%dx%d o %dx%d -> %dx%dx%d [s: %dx%d]',
      from, inj, ini, kj, ki, to, outj, outi, sj, si)
   times[title] = tm
   
   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,1,1)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,1,1):cl()
   gconv.weight = sconv.weight:cl()
   gconv.bias = sconv.bias:cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundtruth
   local err_max = error:abs():max()
   local val_max = groundtruth:max()
   local val_abs_avg = groundtruth:clone():abs():mean()
--   print('err_max', err_max, 'val_max', val_max, 'val_avg', val_abs_avg)
   mytester:assertlt(error:abs():max(), 1e-5, 'error on state (forward) ')
end

function clnntest.SpatialConvolutionMM_forward_batch()
   torch.manualSeed(123)
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = 1 -- not supported by CPU version yet
   local sj = si
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local tm = {}
   local title = string.format('SpatialConvolutionMM.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]',
      bs, from, inj, ini, kj, ki, bs, to, outj, outi, sj, si)
   times[title] = tm
   
   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real
   
   input = input:cl()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cl()
   gconv.weight = sconv.weight:cl()
   gconv.bias = sconv.bias:cl()
   local rescl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundtruth
   mytester:assertlt(error:abs():max(), 1e-5, 'error on state (forward) ')
end

function clnntest.SpatialConvolutionMM_backward_single()
   torch.manualSeed(123)
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = 1 -- not supported by CPU version yet
   local sj = si
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local tm = {}
   local title = string.format('SpatialConvolutionMM.backward %dx%dx%d o %dx%d -> %dx%dx%d',
      from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm
   
   local input = torch.randn(from,inj,ini)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real
   
   input = input:cl()
   gradOutput = gradOutput:cl()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cl()
   gconv.weight = sconv.weight:cl()
   gconv.bias = sconv.bias:cl()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescl = gconv:backward(input, gradOutput)
   end
   local weightcl = gconv.gradWeight
   local biascl = gconv.gradBias
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundgrad
   local werror = weightcl:float() - groundweight
   local berror = biascl:float() - groundbias
   
   mytester:assertlt(error:abs():max(), 1e-5, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), 4e-4, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), 1e-4, 'error on bias (backward) ')
end

function clnntest.SpatialConvolutionMM_backward_batch()
   torch.manualSeed(123)
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = 1 -- not supported by CPU version yet
   local sj = si
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local tm = {}
   local title = string.format('SpatialConvolutionMM.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
      bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm
   
   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real
   
   input = input:cl()
   gradOutput = gradOutput:cl()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj):cl()
   gconv.weight = sconv.weight:cl()
   gconv.bias = sconv.bias:cl()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescl = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescl = gconv:backward(input, gradOutput)
   end
   local weightcl = gconv.gradWeight
   local biascl = gconv.gradBias
   cltorch.synchronize()
   tm.gpu = a:time().real
   
   local error = rescl:float() - groundgrad
   local werror = weightcl:float() - groundweight
   local berror = biascl:float() - groundbias
   
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

