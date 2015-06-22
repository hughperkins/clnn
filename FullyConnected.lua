-- plan is to use like this:
-- model:add(FullyConnected(15)
-- ... where 15 is number of neurons
--
-- this would then take a 4d tensor input(batchSize, inPlanes, inSize, inSize)
--
-- ... and convert it to 4d tensor output(batchSize, numNeurons, 1, 1)

local FullyConnected, parent = torch.class('nn.FullyConnected', 'nn.Sequential')

function FullyConnected:__init(neurons)
  parent.__init(self)
  self.neurons = neurons
end

function FullyConnected:updateOutput(input)
  if #self.modules == 0 then
    if( input:size():size() ~= 4) then
      error("input should be 4d tensor")
    end
    print('FC:update output type(input)', torch.type(input))
    local batchSize = input:size(1)
    local inPlanes = input:size(2)
    local inH = input:size(3)
    local inW = input:size(4)
    local r1 = nn.Reshape(batchSize, inPlanes * inH * inW)
    r1:type(torch.type(input))
--    r1._input = input.__constructor()
--    r1._gradOutput = input.__constructor()
    print('type r1._input', torch.type(r1._input))
    self:add(r1)
    local linear = nn.Linear(inPlanes * inH * inW, self.neurons)
    linear:type(torch.type(input))
    print('type linear.output', torch.type(linear.output))
    self:add(linear)
    self.gradWeight = linear.gradWeight
    self.gradBias = linear.gradBias
    local r2 = nn.Reshape(batchSize, self.neurons, 1, 1)
    r2:type(torch.type(input))
--    r2._input = input.__constructor()
--    r2._gradOutput = input.__constructor()
    print('type r2._input', torch.type(r2._input))
    self:add(r2)
  end
  parent.updateOutput(self, input)
  return self.output
end

