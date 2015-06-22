function torch.ClTensor.nn.LogSoftMax_updateOutput(self, input)
  cltorch.setTrace(1)
  if   input:dim() ~= 2 then
    error('LogSoftMax expects 2-d tensor currently')
  end
  if self.buffer == nil then
    self.buffer = input:clone()
    self.maxbuffer, self.resind = input:max(2)
    self.vec_size = input:size(2)
  end
  self.output:resize(input:size())

  self.maxbuffer:max(self.resind, input, 2)
  self.buffer:repeatTensor(self.maxbuffer, 1, self.vec_size)

  self.output:copy(input)
  self.output:csub(self.buffer)
  self.output:exp()
  self.maxbuffer:sum(self.output,2)
  self.buffer:repeatTensor(self.maxbuffer, 1, self.vec_size)

  self.output:cdiv(self.buffer)
  self.output:log()

  cltorch.setTrace(0)
  return self.output
end

function torch.ClTensor.nn.LogSoftMax_updateGradInput(self, input, gradOutput)
  local nElement = self.gradInput:nElement()
  self.gradInput:resizeAs(input)
  if self.gradInput:nElement() ~= nElement then
     self.gradInput:zero()
  end

  cltorch.setTrace(1)
  self.maxbuffer:sum(gradOutput, 2)
  self.buffer:repeatTensor(self.maxbuffer, 1, self.vec_size)
  self.gradInput:copy(self.output)
  self.gradInput:exp()
  self.gradInput:cmul(self.buffer)
  self.gradInput:neg()
  self.gradInput:add(gradOutput)

  cltorch.setTrace(0)
  return self.gradInput
end


