require 'nn'

nn.SpatialMaxPooling.baseUpdateOutput = nn.SpatialMaxPooling.updateOutput
nn.SpatialMaxPooling.baseUpdateGradInput = nn.SpatialMaxPooling.updateGradInput

function nn.SpatialMaxPooling:updateOutput(input)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateOutput(input)
   end

   input.nn.SpatialMaxPooling_updateOutput(self, input)
   return self.output
end

function nn.SpatialMaxPooling:updateGradInput(input, gradOutput)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateGradInput(input, gradOutput)
   end

   input.nn.SpatialMaxPooling_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

