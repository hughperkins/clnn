-- as you can see, you will need to 'luarocks install mnist' first

require 'nn'
require 'clnn'
local mnist = require 'mnist'

local _trainset = mnist.traindataset()
local _testset = mnist.testdataset()

local trainset = {}
function trainset.size()
  return 1280
end
local train_data = _trainset.data
local train_labels = _trainset.label
for i=1,1280 do
  local e = {}
  e[1] = torch.reshape(train_data[i], 28 * 28):double():cl()
  local expectedout = torch.Tensor(10)
  expectedout:zero()
  local label = train_labels[i]
  if label == 0 then
    label = 10
  end
  expectedout[label] = 1
  e[2] = expectedout:cl()
  trainset[#trainset+1] = e
end

e1 = trainset[1]

print('Ntrain', trainset.size())

local net = nn.Linear(28 * 28, 10):cl()
print('net\n', net)
local criterion = nn.MSECriterion():cl()
local trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.000001
print('learningRate', trainer.learningRate)
trainer:train(trainset)


