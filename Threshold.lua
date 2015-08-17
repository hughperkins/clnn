require 'nn'
require 'cltorch'

local function floatToString(val)
   local valstring = tostring(val)
   if valstring:find('%.') or valstring:find('e') then
      valstring = valstring .. 'f'
   end
   return valstring
end

function torch.ClTensor.nn.Threshold_updateOutput(self, input)
   --  print('torch.nn.ReLU
   --  self.output:map(input, "*out = *in1 > 0.0f ? *in1 : 0.0f")
   self.thresholdstring = floatToString(self.threshold)
   self.valstring = floatToString(self.val)
   if self.inplace then
      input:apply("*out = (*out > " .. self.thresholdstring .. ") ? *out : " .. self.valstring)
      self.output = input
   else
      self.output:resize(input:size())
      self.output:map(input, "*out = ( *in1 > " .. self.thresholdstring .. ") ? *in1 : " .. self.valstring)
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
      gradOutput:map2(input, gradOutput, "*out = (*in1 > " .. self.thresholdstring .. ") ? *in2 : 0.0f")
      self.gradInput = gradOutput
   else
      self.gradInput:map2(input, gradOutput, "*out = (*in1 > " .. self.thresholdstring .. ") ? *in2 : 0.0f")
   end
   --  self.gradInput:map2(gradOutput, self.output, "*out = *in2 > 0.0f ? *in1 : 0.0f")
   return self.gradInput
end

