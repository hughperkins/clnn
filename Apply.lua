-- this will take arbitrary number of inputs and outputs
-- presumably as tables
-- and apply an opencl expression to them, per-element
-- all inputs and outputs should have the same size
-- note that the outputs are *not* identical

local Apply, parent = torch.class('nn.Apply', 'nn.Module')

function Apply:__init(numInputs, numOutputs, forwardExpression, backwardExpression)
  parent.__init(self)
  self.numInputs = numInputs
  self.numOutputs = numOutputs
  self.forwardSrc = [[
    int n = get_global_id(0);
    if(n >= N) {
      return;
    }
  ]]
  fe = forwardExpression
  for i=1,numInputs do
    fe = fe:gsub('{{in' .. i .. '}}', 'in' .. i .. '_data[n]')
  end
  for i=1,numOutputs do
    fe = fe:gsub('{{out' .. i .. '}}', 'out' .. i .. '_data[n]')
  end
  self.forwardSrc = self.forwardSrc .. fe
  self.backwardExpression = backwardExpression
  inputs = {}
  for i=1,numInputs do
    inputs['in' .. i] = 'ClTensor'
  end
  inputs['N'] = 'int'
  outputs = {}
  for i=1,numOutputs do
    outputs['out' .. i] = 'ClTensor'
  end
  self.forwardKernel = torch.ClKernel({input=inputs, output=outputs, src=self.forwardSrc})
  print(self.forwardKernel, self.forwardKernel:getRawKernel(), self.forwardKernel:getRenderedKernel())
end

-- input should be a table of inputs
-- output will be a table of outputs
function Apply:updateOutput(inputs)
  if #inputs ~= self.numInputs then
    error("num inputs should be " .. self.numInputs)
  end
  self.outputs = self.outputs or {}
  for o=1,self.numOutputs do
    self.outputs[o] = self.outputs[o] or inputs[1].new()
    self.outputs[o]:resizeAs(inputs[1])
  end
  forwardParams = {}
  for i=1,self.numInputs do
    forwardParams['in' .. i] = inputs[i]
  end
  for o=1,self.numOutputs do
    forwardParams['out' .. o] = self.outputs[o]
  end
  forwardParams['N'] = inputs[1]:numel()
  self.forwardKernel:run(forwardParams)
  self.output = self.outputs
  return self.outputs
end

function Apply:updateGradInput(input, gradOutput)
  return input -- useless nop, for now
end

