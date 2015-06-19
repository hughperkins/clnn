function torch.ClTensor.nn.Tanh_updateOutput(self, input)
  self.output = torch.tanh(input)
  return self.output
end

function torch.ClTensor.nn.Tanh_updateGradInput(self, input, gradOutput)
  local nElement = self.gradInput:nElement()
  self.gradInput:resizeAs(input)
  if self.gradInput:nElement() ~= nElement then
     self.gradInput:zero()
  end
  self.gradInput:map2(gradOutput, self.output, "*out = *in1 * (1 - *in2 * *in2)")
--  self.gradInput:copy(self.output)
--  self.gradInput:pow(2)
--  self.gradInput:neg()
--  self.gradInput:add(1)
--  self.gradInput:cmul(gradOutput)
 -- self.gradInput = torch.cmul(gradOutput, - torch.pow(self.output, 2) + 1)
  return self.gradInput
end

