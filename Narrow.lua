require 'nn'

-- monkey patch it in :-P

-- this is basically copied from nn.Narrow
-- but we dont zero out whole gradInput, except the first time

--nn.Narrow.baseUpdateOutput = nn.Narrow.updateOutput
nn.Narrow.baseUpdateGradInput = nn.Narrow.updateGradInput

--function nn.Narrow:updateOutput(input, target)
--  if torch.type(input) ~= 'torch.ClTensor' then
--    return self:baseUpdateOutput(input, target)
--  end
--  return self.output
--end


function nn.Narrow:updateGradInput(input, gradOutput)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateGradInput(input, gradOutput)
   end
   if self._doneinit == nil then
      --    print('resizing gradinput')
      self.gradInput:resizeAs(input)
      self.gradInput:zero();
      self._doneinit = true
   end
   
   --  self.gradInput:resizeAs(input)
   --  self.gradInput:zero();
   self.gradInput:narrow(self.dimension,self.index,self.length):copy(gradOutput)
   
   return self.gradInput
end

