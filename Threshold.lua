require 'nn'
require 'cltorch'

function torch.ClTensor.nn.Threshold_updateOutput(self, input)
--  print('torch.nn.ReLU
--  self.output:map(input, "*out = *in1 > 0.0f ? *in1 : 0.0f")
  if self.inplace then
    input:apply("*out = (*out > " .. self.threshold .. ") ? *out : " .. self.val)
    self.output = input
  else
    self.output:resize(input:size())
    self.output:map(input, "*out = ( *in1 > " .. self.threshold .. ") ? *in1 : " .. self.val)
  end
  return self.output
end

function torch.ClTensor.nn.Threshold_updateGradInput(self, input, gradOutput)
  local nElement = self.gradInput:nElement()
  self.gradInput:resizeAs(input)
  if self.gradInput:nElement() ~= nElement then
     self.gradInput:zero()
  end
  if self.inplace then
    gradOutput:map2(input, gradOutput, "*out = (*in1 > " .. self.threshold .. ") ? *in2 : 0")
    self.gradInput = gradOutput
  else
    self.gradInput:map2(input, gradOutput, "*out = (*in1 > " .. self.threshold .. ") ? *in2 : 0")
  end
--  self.gradInput:map2(gradOutput, self.output, "*out = *in2 > 0.0f ? *in1 : 0.0f")
  return self.gradInput
end


