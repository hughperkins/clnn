-- as you can see, you will need to 'luarocks install mnist' first

require 'nn'
require 'clnn'
local mnist = require 'mnist'

local _trainset = mnist.traindataset()
local _testset = mnist.testdataset()

--function trainset.size()
--  return 1280
--end

--function testset.size()
--  return 1280
--end

--trainset:size(1280)

--trainset.size = 1280
--testset.size = 1280

local trainset = {}
function trainset.size()
  return 1280
end
local train_data = _trainset.data
local train_labels = _trainset.label
for i=1,1280 do
-- for i,row in ipairs(_trainset)
--  print(torch.reshape(train_data[i], 28 *28))
--  print(train_labels[i])
  local e = {}
  e[1] = torch.reshape(train_data[i], 28 * 28):double()
  local expectedout = torch.Tensor(10)
--  print('expectedout\n', expectedout)
  expectedout:zero()
--  print('expectedout\n', expectedout)
  local label = train_labels[i]
  if label == 0 then
    label = 10
  end
--  print('label', label)
  expectedout[label] = 1
--  e[2] = torch.Tensor({train_labels[i]})
  e[2] = expectedout
  trainset[#trainset+1] = e
end

e1 = trainset[1]
print('e1[1]', e1[1])
print('e1[2]', e1[2])

print('Ntrain', trainset.size()) -- to retrieve the size
--print('Ntest', testset.size()) -- to retrieve the size

local net = nn.Linear(28 * 28, 10)
local criterion = nn.MSECriterion() -- Mean Squared Error criterion
local trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.01
print('learningRate', trainer.learningRate)
trainer:train(trainset) -- train using some examples


