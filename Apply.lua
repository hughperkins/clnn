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

  self.forwardExpression = forwardExpression:gsub('{{input}}', '{{input1}}')
  self.forwardExpression = self.forwardExpression:gsub('{{output}}', '{{output1}}')

  self.backwardExpression = backwardExpression

--  print('self', self)
--  Apply.updateExpressions(self, numInputs, numOutputs, forwardExpression, backwardExpression)
end

function Apply.updateExpressions(self, numInputs, numOutputs, forwardExpression, backwardExpression)
  self.numInputs = numInputs
  self.numOutputs = numOutputs

  -- create forward kernel
  self.forwardExpression = forwardExpression:gsub('{{input}}', '{{input1}}')
  self.forwardExpression = self.forwardExpression:gsub('{{output}}', '{{output1}}')
  self.forwardSrc = [[
    int n = get_global_id(0);
    if(n >= N) {
      return;
    }
  ]]
  fe = forwardExpression
  for i=1,numInputs do
    fe = fe:gsub('{{input' .. i .. '}}', 'input' .. i .. '_data[n]')
  end
  fe = fe:gsub('{{input}}', 'input1_data[n]')
  for i=1,numOutputs do
    fe = fe:gsub('{{output' .. i .. '}}', 'output' .. i .. '_data[n]')
  end
  fe = fe:gsub('{{output}}', 'output1_data[n]')
  self.forwardSrc = self.forwardSrc .. fe
  self.backwardExpression = backwardExpression
  local inputs = {}
  for i=1,numInputs do
    inputs['input' .. i] = 'ClTensor'
  end
  inputs['N'] = 'int'
  local outputs = {}
  for i=1,numOutputs do
    outputs['output' .. i] = 'ClTensor'
  end
  self.forwardKernel = torch.ClKernel({input=inputs, output=outputs, src=self.forwardSrc, name='updateOutput'})
--  print(self.forwardKernel, self.forwardKernel:getRawKernel(), self.forwardKernel:getRenderedKernel())

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
--    be = be:gsub('{{input' .. i .. '}}', 'input' .. i .. '_data[n]')
    be = be:gsub('{{gradInput' .. i .. '}}', 'gradInput' .. i .. '_data[n]')
    be = be:gsub('{{input' .. i .. '}}', 'input' .. i .. '_data[n]')
  end
  be = be:gsub('{{input}}', 'input1_data[n]')
  be = be:gsub('{{gradInput}}', 'gradInput1_data[n]')
  for o=1,numOutputs do
    be = be:gsub('{{gradOutput' .. o .. '}}', 'gradOutput' .. o .. '_data[n]')
    be = be:gsub('{{output' .. o .. '}}', 'output' .. o .. '_data[n]')
  end
  be = be:gsub('{{gradOutput}}', 'gradOutput1_data[n]')
  be = be:gsub('{{output}}', 'output1_data[n]')
  self.backwardSrc = self.backwardSrc .. be
  self.backwardExpression = backwardExpression
  local inputs = {}  -- this is certainly gratuitously duplicated
  local outputs = {}
  for i=1,numInputs do
--    inputs['input' .. i] = 'ClTensor'
    outputs['gradInput' .. i] = 'ClTensor'
    inputs['input' .. i] = 'ClTensor'
  end
  inputs['N'] = 'int'
  for i=1,numOutputs do
--    inputs['output' .. i] = 'ClTensor'
    inputs['gradOutput' .. i] = 'ClTensor'
  end
  self.backwardKernel = torch.ClKernel({input=inputs, output=outputs, src=self.backwardSrc, name='updateGradInput'})
--  print(self.backwardKernel, self.backwardKernel:getRawKernel(), self.backwardKernel:getRenderedKernel())
end

function Apply:__tostring()
  local fe = ''
  if self.forwardExpression ~= nil then
    fe = self.forwardExpression:gsub('\n', ''):gsub('  ', '')
  end
  return 'Apply(' .. tostring(self.numInputs) .. ', ' .. tostring(self.numOutputs) .. ', "' .. fe .. '")'
end

-- input should be a table of inputs
-- output will be a table of outputs
function Apply:updateOutput(inputs)
  if torch.type(inputs) == 'torch.ClTensor' then
    inputs = {inputs}
  end
  print('torch.type(inputs)', torch.type(inputs), #inputs)
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
    forwardParams['input' .. i] = inputs[i]
  end
  for o=1,self.numOutputs do
    forwardParams['output' .. o] = self.outputs[o]
  end
  forwardParams['N'] = inputs[1]:numel()
  self.forwardKernel:run(forwardParams)
  if self.numOutputs == 1 then
    self.outputs = self.outputs[1]
  end
  self.output = self.outputs
  return self.outputs
end

function Apply:updateGradInput(inputs, gradOutputs)
  print('Apply:updateGradInput')
  if torch.type(inputs) == 'torch.ClTensor' then
    inputs = {inputs}
  end
  if torch.type(gradOutputs) == 'torch.ClTensor' then
    gradOutputs = {gradOutputs}
  end
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
--    backwardParams['input' .. i] = inputs[i]
    backwardParams['gradInput' .. i] = self.gradInputs[i]
  end
  for o=1,self.numOutputs do
    backwardParams['gradOutput' .. o] = gradOutputs[o]
    backwardParams['output' .. o] = self.outputs[o]
  end
  backwardParams['N'] = inputs[1]:numel()
  self.backwardKernel:run(backwardParams)
  if self.numInputs == 1 then
    self.gradInputs = self.gradInputs[1]
  end
  self.gradInput = self.gradInputs
  return self.gradInputs
end

