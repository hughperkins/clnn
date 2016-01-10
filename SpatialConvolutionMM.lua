require 'nn'

nn.SpatialConvolutionMM.baseUpdateOutput = nn.SpatialConvolutionMM.updateOutput
nn.SpatialConvolutionMM.baseUpdateGradInput = nn.SpatialConvolutionMM.updateGradInput
nn.SpatialConvolutionMM.baseAccGradParameters = nn.SpatialConvolutionMM.accGradParameters

function nn.SpatialConvolutionMM:updateOutput(input)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateOutput(input)
   end

   input.nn.SpatialConvolutionMM_updateOutput(self, input)
   return self.output
end

function nn.SpatialConvolutionMM:updateGradInput(input, gradOutput)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateGradInput(input, gradOutput)
   end

   input.nn.SpatialConvolutionMM_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function nn.SpatialConvolutionMM:accGradParameters(input, gradOutput, scale)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseAccGradParameters(input, gradOutput, scale)
   end

   input.nn.SpatialConvolutionMM_accGradParameters(self, input, gradOutput, scale)
   return self.gradInput
end

