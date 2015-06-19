function torch.ClTensor.nn.ReLU_updateOutput(self, input)
  self.output:map(input, "*out = *in1 > 0.0f ? *in1 : 0.0f")
  return self.output
end

function torch.ClTensor.nn.ReLU_updateGradInput(self, input, gradOutput)
  local nElement = self.gradInput:nElement()
  self.gradInput:resizeAs(input)
  if self.gradInput:nElement() ~= nElement then
     self.gradInput:zero()
  end
  self.gradInput:map2(gradOutput, self.output, "*out = *in2 > 0.0f ? *in1 : 0.0f")
  return self.gradInput
end

