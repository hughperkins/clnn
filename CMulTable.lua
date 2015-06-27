require 'nn'

-- monkey patch it in :-P

nn.CMulTable.baseUpdateOutput = nn.CMulTable.updateOutput
nn.CMulTable.baseUpdateGradInput = nn.CMulTable.updateGradInput

function nn.CMulTable:updateOutput(input, target)
  if torch.type(input[1]) ~= 'torch.ClTensor' then
    return self:baseUpdateOutput(input, target)
  end 
  -- copied from nn for now
   self.output:resizeAs(input[1])
if true then
  if #input == 2 then  -- this is really common case
                       -- anyway, it's what char-rnn uses :-P
    self.output:cmul(input[1], input[2])
  else
     self.output:copy(input[1])
     for i=2,#input do
        self.output:cmul(input[i])
     end
  end
end
   return self.output
end


function nn.CMulTable:updateGradInput(input, gradOutput)
--  print('cl.updategradinput', torch.type(input), torch.type(gradOutput))
  if torch.type(gradOutput) ~= 'torch.ClTensor' then
    return self:baseUpdateGradInput(input, gradOutput)
  end
  -- copied from nn for now
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:resizeAs(input[i])
if true then
      if #input == 2 then
        self.gradInput[i]:cmul(gradOutput, input[2-i + 1])
      else
        self.gradInput[i]:copy(gradOutput)
        for j=1,#input do
           if i~=j then
              self.gradInput[i]:cmul(input[j])
           end
        end
      end
end
   end
   return self.gradInput
end

