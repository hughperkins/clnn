-- as you can see, you will need to 'luarocks install mnist' first

require 'nn'
require 'sys'
local mnist = require 'mnist'

api = os.getenv('API')
local batchSize = 128
local numBatches = 40
learningRate = 0.1
maxIteration = 7

netchoice = os.getenv('MODEL')
if netchoice == nil then
  netchoice = 'linear'
end

if api == nil then
  api = 'cl'
--  print('Please set the "API" env var to your choice of api, being one of: cpu, cl, cuda')
--  print('eg, run like:')
--  print('  API=cpu ./run-mnist2.sh')
--  print('  API=cuda ./run-mnist2.sh')
--  print('  API=cl ./run-mnist2.sh')
end

if api == 'cuda' then
  require 'cunn'
end
--cltorch.setTrace(1)

local _trainset = mnist.traindataset()
--local _testset = mnist.testdataset()

local train_data = _trainset.data
local train_labels = _trainset.label

local trainset = {}
function trainset.size()
  return numBatches
end

i = 1
for b=1,numBatches do
  local batch_data = torch.Tensor(batchSize, 28*28)
  batch_data = batch_data / 255 - 0.5
  local batch_labels = torch.Tensor(batchSize, 10)
  for bi=1,batchSize do
    batch_data[bi] = torch.reshape(train_data[i], 28 * 28):double()
    batch_labels[bi]:zero()
    local label = train_labels[i]
    if label == 0 then
      label = 10
    end
    batch_labels[bi][label] = 1
    i = i + 1
  end
  table.insert(trainset, {batch_data, batch_labels})
end

local model = nn.Sequential()
if netchoice == 'linear' then
  model:add(nn.Linear(28*28,150))
  model:add(nn.Tanh())
  model:add(nn.Linear(150,10))
elseif netchoice == 'conv1' then
  model:add(nn.Reshape(128, 1, 28, 28))
  -- from https://github.com/torch/demos/blob/4abff87d89f7ad8de3c51cbd0fe549b6000f3a1a/train-a-digit-classifier/train-on-mnist.lua#L80
  model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
  model.modules[#model].padW = model.modules[#model].padding
  model.modules[#model].padH = model.modules[#model].padding
--  model[#model].padH = model[#model].padding
  model:add(nn.Tanh())
  model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
  model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
  model.modules[#model].padW = model.modules[#model].padding
  model.modules[#model].padH = model.modules[#model].padding
  print('model.modules[#model].padding',  model.modules[#model].padding)
  model:add(nn.Tanh())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.Reshape(64*2*2))
  model:add(nn.Linear(64*2*2, 200))
  model:add(nn.Tanh())
  model:add(nn.Linear(200, 10))
else
  error('net choice not recongized', netchoice)
end

if api == 'cpu' then
  local criterion = nn.MSECriterion()
  local trainer = nn.StochasticGradient(model, criterion)
  trainer.maxIteration = maxIteration
  trainer.learningRate = learningRate
  sys.tic()
  trainer:train(trainset)
  print('toc', sys.toc())
end

if api == 'cl' then
  require 'cltorch'
  require 'clnn'
--  cltorch.setTrace(1)
  local trainsetcl = {}
  function trainsetcl.size()
    return numBatches
  end
  for b=1,numBatches do
    table.insert(trainsetcl, {trainset[b][1]:clone():cl(), trainset[b][2]:clone():cl()})
  end
  local modelcl = model:cl()
  local criterioncl = nn.MSECriterion():cl()
  local trainercl = nn.StochasticGradient(modelcl, criterioncl)
  trainercl.maxIteration = 1
  trainercl.learningRate = learningRate
  for it=1,maxIteration do
    sys.tic()
    trainercl:train(trainsetcl)
    print('toc', sys.toc())
  end
end

if api == 'cuda' then
  require 'cunn'
  local trainsetcuda = {}
  function trainsetcuda.size()
    return numBatches
  end
  for b=1,numBatches do
    table.insert(trainsetcuda, {trainset[b][1]:clone():cuda(), trainset[b][2]:clone():cuda()})
  end
  local modelcuda = model:cuda()
  local criterioncuda = nn.MSECriterion():cuda()
  local trainercuda = nn.StochasticGradient(modelcuda, criterioncuda)
  trainercuda.maxIteration = 1
  trainercuda.learningRate = learningRate
  for it=1,maxIteration do
    sys.tic()
    trainercuda:train(trainsetcuda)
    print('toc', sys.toc())
  end
end


