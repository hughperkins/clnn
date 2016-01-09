require 'nn'

nn.ELU.baseUpdateOutput = nn.ELU.updateOutput
nn.ELU.baseUpdateGradInput = nn.ELU.updateGradInput

function nn.ELU:updateOutput(input)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateOutput(input)
   end

   input.nn.ELU_updateOutput(self, input)
   return self.output
end

function nn.ELU:updateGradInput(input, gradOutput)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateGradInput(input, gradOutput)
   end

   input.nn.ELU_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

