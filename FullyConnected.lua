-- FullyConnected layer
-- ====================
--
--    model:add(nn.FullyConnected(numNeurons))
--
-- where:
--   numNeurons: number of neurons (can be any positive integer,
--               within the bounds of available memory and other
--               resources)
--
-- input:
--   4d tensor input(batchSize, inPlanes, inHeight, inWidth)
--
-- output:
--   4d tensor output(batchSize, numNeurons, 1, 1)
--
-- FullyConnected wraps a Linear layer inside two Reshape layers,
-- see the updateOutput method below for more details

local FullyConnected, parent = torch.class('nn.FullyConnected', 'nn.Sequential')

function FullyConnected:__init(neurons)
  parent.__init(self)
  self.neurons = neurons
end

function FullyConnected:__tostring__()
  return 'nn.FullyConnected(' .. self.neurons .. ')'
end

function FullyConnected:updateOutput(input)
  if #self.modules == 0 then
    if( input:size():size() ~= 4) then
      error("input should be 4d tensor")
    end
    local batchSize = input:size(1)
    local inPlanes = input:size(2)
    local inH = input:size(3)
    local inW = input:size(4)
    local r1 = nn.Reshape(batchSize, inPlanes * inH * inW)
    r1:type(torch.type(input))
    self:add(r1)
    local linear = nn.Linear(inPlanes * inH * inW, self.neurons)
    linear:type(torch.type(input))
    self:add(linear)
    self.gradWeight = linear.gradWeight
    self.gradBias = linear.gradBias
    local r2 = nn.Reshape(batchSize, self.neurons, 1, 1)
    r2:type(torch.type(input))
    self:add(r2)
  end
  parent.updateOutput(self, input)
  return self.output
end

