require 'nn'

nn.SpatialAveragePooling.baseUpdateOutput = nn.SpatialAveragePooling.updateOutput
nn.SpatialAveragePooling.baseUpdateGradInput = nn.SpatialAveragePooling.updateGradInput

function nn.SpatialAveragePooling:updateOutput(input)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateOutput(input)
   end

   input.nn.SpatialAveragePooling_updateOutput(self, input)
   return self.output
end

function nn.SpatialAveragePooling:updateGradInput(input, gradOutput)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateGradInput(input, gradOutput)
   end

   input.nn.SpatialAveragePooling_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

