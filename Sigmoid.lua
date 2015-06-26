function torch.ClTensor.nn.Sigmoid_updateOutput(self, input)
--  if self ~= nil then
--    print('sigmoid self size', self:size())
--  else
--   print('self nil')
--  end
  self.output:resize(input:size())
--  print('sigmoid input size', input:size())
  self.output:map(input, "*out = 1.0f / (1.0f + exp( - *in1))")
  return self.output
end

function torch.ClTensor.nn.Sigmoid_updateGradInput(self, input, gradOutput)
  local nElement = self.gradInput:nElement()
  self.gradInput:resizeAs(input)
  if self.gradInput:nElement() ~= nElement then
     self.gradInput:zero()
  end
  self.gradInput:map2(gradOutput, self.output, "*out = *in1 * *in2 * (1 - *in2)")
  return self.gradInput
end

