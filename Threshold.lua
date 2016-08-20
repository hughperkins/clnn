require 'nn'
require 'cltorch'

nn.Threshold.baseUpdateOutput = nn.Threshold.updateOutput
nn.Threshold.baseUpdateGradInput = nn.Threshold.updateGradInput

local function floatToString(val)
   local valstring = tostring(val)
   if valstring:find('%.') or valstring:find('e') then
      valstring = valstring .. 'f'
   end
   return valstring
end

function nn.Threshold.updateOutput(self, input)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateOutput(input)
   end

   self.thresholdstring = floatToString(self.threshold)
   self.valstring = floatToString(self.val)
   if self.inplace then
      input:apply_on_gpu("*out = (*out > " .. self.thresholdstring .. ") ? *out : " .. self.valstring)
      self.output = input
   else
      self.output:resize(input:size())
      self.output:map_on_gpu(input, "*out = ( *in1 > " .. self.thresholdstring .. ") ? *in1 : " .. self.valstring)
   end
   return self.output
end

function nn.Threshold.updateGradInput(self, input, gradOutput)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateGradInput(input, gradOutput)
   end

   local nElement = self.gradInput:nElement()
   self.gradInput:resizeAs(input)
   if self.gradInput:nElement() ~= nElement then
      self.gradInput:zero()
   end
   if self.inplace then
      gradOutput:map2_on_gpu(input, gradOutput, "*out = (*in1 > " .. self.thresholdstring .. ") ? *in2 : 0.0f")
      self.gradInput = gradOutput
   else
      self.gradInput:map2_on_gpu(input, gradOutput, "*out = (*in1 > " .. self.thresholdstring .. ") ? *in2 : 0.0f")
   end
   return self.gradInput
end

