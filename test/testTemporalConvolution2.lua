local _test = clnn._test
local times = _test.times
local clnntest = _test.clnntest
local x_clnntest = _test.x_clnntest
local nloop = _test.nloop
local precision_forward = 1e-4
local precision_backward = 1e-3

require 'optim'

function clnntest.TemporalConvolution2_forward()  -- check forward 'by hand', then use optim for backward
  local batchSize = 5
  local inFeatures = 6
  local outFeatures = 7
  local sentenceLength = 10
  local kernelSize = 3
--  local stride = 1
  local input = torch.ClTensor(batchSize, sentenceLength, inFeatures):uniform()
  local net = nn.TemporalConvolution2(inFeatures, outFeatures, kernelSize)
  net:cl()
  local weights = net.weight
  weights:uniform(-1.0, 1.0)
  net.bias:zero()  -- simplify test for now...
  local output = net:forward(input)
  -- calc 'by hand' to check
--  print('weights:size()', weights:size())
--  print('output:size()', output:size())
  local outLength = sentenceLength - math.floor(kernelSize / 2) * 2
  local ourOut = torch.FloatTensor(batchSize, outLength, outFeatures):zero()
  -- each batch item is independent, calculated separately from others
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
            local weight = weights[outFeature][inFeature][k][1]
            sum = sum + weight * input[b][inS][inFeature]
          end
        end
        ourOut[b][outS][outFeature] = sum
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

function clnntest.TemporalConvolution2_backward_gradInput()
  torch.manualSeed(123)
  local batchSize = 5
  local inFeatures = 6
  local outFeatures = 7
  local sentenceLength = 10
  local kernelSize = 3
  local input = torch.ClTensor(batchSize, sentenceLength, inFeatures):uniform()
  local net = nn.TemporalConvolution2(inFeatures, outFeatures, kernelSize)
  net:cl()
  local params, gradParams = net:getParameters()

  local target = torch.ClTensor(batchSize, sentenceLength - 2, outFeatures):uniform()
  local crit = nn.MSECriterion()
  crit:cl()

  function opfunc(collapsedInput)
    net:evaluate()
    local out = net:forward(input)
    local loss = crit:forward(out, target)
    local gradOutput = crit:backward(out, target)
    local gradInput = net:backward(input, gradOutput)
    return loss, gradInput:view(-1):double()
  end

  local eps = 1e-2
  local d, c, c_est = optim.checkgrad(opfunc, input:view(-1), eps)
--  local comp = torch.FloatTensor(c:size(1), 2)
--  comp:narrow(2,1,1):copy(c)
--  comp:narrow(2,2,1):copy(c_est)
--  print('comp', comp)
--  print('d', d)
  mytester:assertlt(math.abs(d), 0.001)
end

function clnntest.TemporalConvolution2_backward_gradParams()
  local batchSize = 5
  local inFeatures = 6
  local outFeatures = 7
  local sentenceLength = 10
  local kernelSize = 3
  local input = torch.ClTensor(batchSize, sentenceLength, inFeatures):uniform()
  local net = nn.TemporalConvolution2(inFeatures, outFeatures, kernelSize)
  net:cl()
  local params, gradParams = net:getParameters()

  local target = torch.ClTensor(batchSize, sentenceLength - 2, outFeatures):uniform()
  local crit = nn.MSECriterion()
  crit:cl()

  function opfunc(params)
    net:evaluate()
    gradParams:zero()
    local out = net:forward(input)
    local loss = crit:forward(out, target)
    local gradOutput = crit:backward(out, target)
    local gradInput = net:backward(input, gradOutput)
    return loss, gradParams:double()
  end

  local eps = 1e-2
  local d, c, c_est = optim.checkgrad(opfunc, params, eps)
--  local comp = torch.FloatTensor(c:size(1), 2)
--  comp:narrow(2,1,1):copy(c)
--  comp:narrow(2,2,1):copy(c_est)
--  print('comp', comp)
--  print('d', d)
  mytester:assertlt(math.abs(d), 0.001)
end

