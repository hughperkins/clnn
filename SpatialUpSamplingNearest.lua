require 'nn'

nn.SpatialUpSamplingNearest.baseUpdateOutput = nn.SpatialUpSamplingNearest.updateOutput
nn.SpatialUpSamplingNearest.baseUpdateGradInput = nn.SpatialUpSamplingNearest.updateGradInput

function nn.SpatialUpSamplingNearest:updateOutput(input)
   if torch.type(input) ~= 'torch.ClTensor' then
     return self:baseUpdateOutput(input, target)
   end
   if input:dim() ~= 4 and input:dim() ~= 3 then
     error('SpatialUpSamplingNearest only support 3D or 4D tensors')
   end
   -- Copy the input size
   local xdim = input:dim()
   local ydim = input:dim() - 1
   for i = 1, input:dim() do
     self.inputSize[i] = input:size(i)
     self.outputSize[i] = input:size(i)
   end
   self.outputSize[ydim] = self.outputSize[ydim] * self.scale_factor
   self.outputSize[xdim] = self.outputSize[xdim] * self.scale_factor
   -- Resize the output if needed
   if input:dim() == 3 then
     self.output:resize(self.outputSize[1], self.outputSize[2],
       self.outputSize[3])
   else
     self.output:resize(self.outputSize)
   end
   input.THNN.SpatialUpSamplingNearest_updateOutput(input:cdata(), self.output:cdata(), self.scale_factor)

   return self.output
end

function nn.SpatialUpSamplingNearest:updateGradInput(input, gradOutput)
    if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateGradInput(input, gradOutput)
    end
   self.gradInput:resizeAs(input)
   input.THNN.SpatialUpSamplingNearest_updateGradInput(input:cdata(), gradOutput:cdata(), self.gradInput:cdata(), self.scale_factor)
   return self.gradInput
end
