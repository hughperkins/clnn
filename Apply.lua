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

  -- create forward kernel
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
  local inputs = {}
  for i=1,numInputs do
    inputs['in' .. i] = 'ClTensor'
  end
  inputs['N'] = 'int'
  local outputs = {}
  for i=1,numOutputs do
    outputs['out' .. i] = 'ClTensor'
  end
  self.forwardKernel = torch.ClKernel({input=inputs, output=outputs, src=self.forwardSrc})
  print(self.forwardKernel, self.forwardKernel:getRawKernel(), self.forwardKernel:getRenderedKernel())

  -- create backward kernel, could probably factorize this, rather than copy and hack
  -- but lets do copy and hack for now, to get it working
  self.backwardSrc = [[
    int n = get_global_id(0);
    if(n >= N) {
      return;
    }
  ]]
  be = backwardExpression
  for i=1,numInputs do
    be = be:gsub('{{in' .. i .. '}}', 'in' .. i .. '_data[n]')
  end
  for i=1,numOutputs do
    be = be:gsub('{{out' .. i .. '}}', 'out' .. i .. '_data[n]')
  end
  self.backwardSrc = self.backwardSrc .. be
  self.backwardExpression = backwardExpression
  local inputs = {}  -- this is certainly gratuitously duplicated
  for i=1,numInputs do
    inputs['in' .. i] = 'ClTensor'
  end
  local outputs = {}
  for i=1,numOutputs do
    outputs['out' .. i] = 'ClTensor'
  end
  outputs['N'] = 'int'
  -- note: need to reverse input/output:
  self.backwardKernel = torch.ClKernel({output=inputs, input=outputs, src=self.backwardSrc})
  print(self.backwardKernel, self.backwardKernel:getRawKernel(), self.backwardKernel:getRenderedKernel())
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
    self.outputs[o]:resizeAs(inputs[1]) -- will be same size, since Apply is strictly per-element, no Narrows and stuff
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

function Apply:updateGradInput(inputs, gradOutputs)
  print('Apply:updateGradInput')
  if #inputs ~= self.numInputs then
    error("num inputs should be " .. self.numInputs)
  end
  if #gradOutputs ~= self.numOutputs then
    error("num gradOutputs should be " .. self.numGradOutputs)
  end
  self.gradInputs = self.gradInputs or {}
  for i=1,self.numInputs do
    self.gradInputs[i] = self.gradInputs[i] or inputs[1].new()
    self.gradInputs[i]:resizeAs(inputs[1])
  end
  backwardParams = {}
  for i=1,self.numInputs do
    backwardParams['in' .. i] = self.gradInputs[i]
  end
  for o=1,self.numOutputs do
    backwardParams['out' .. o] = gradOutputs[o]
  end
  backwardParams['N'] = inputs[1]:numel()
  self.backwardKernel:run(backwardParams)
  self.gradInput = self.gradInputs
  return self.gradInputs
end

