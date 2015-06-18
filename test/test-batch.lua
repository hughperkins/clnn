require 'nn'
require 'clnn'

--data_batch = {}
--table.insert(data_batch, torch.Tensor{0})
--table.insert(data_batch, torch.Tensor{1})

--labels_batch = {}
--table.insert(labels_batch, torch.Tensor{1})
--table.insert(labels_batch, torch.Tensor{3})

--batched_dataset = {}
--function batched_dataset.size()
--  return 1
--end
--table.insert(batched_dataset, {data_batch, labels_batch})

data_batch = torch.Tensor{{0},{1}}
print(data_batch)

labels_batch = torch.Tensor{{1},{3}}
print(labels_batch)

dataset = {}
function dataset.size()
  return 1
end
table.insert(dataset, {data_batch, labels_batch})

net = nn.Linear(1,1)
criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.1  
print('learningRate', trainer.learningRate)
trainer:train(dataset)

