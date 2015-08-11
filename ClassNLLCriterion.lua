require 'nn'

-- monkey patch it in :-P

nn.ClassNLLCriterion.baseUpdateOutput = nn.ClassNLLCriterion.updateOutput
nn.ClassNLLCriterion.baseUpdateGradInput = nn.ClassNLLCriterion.updateGradInput

-- reminder to self: we assume 2-dimensional input
-- dimension 1 is the sampel number, and is the row number
-- dimension 2 is the value for each of the output neurons for that sample
-- we're going to reduce dimension 2 to a size of 1, for each sample,
-- using the knowledge of which is the ground truth class for taht
-- sample.  ie, the loss for that sample is minus the input value
-- for whichever neuron matches the ground truth class
function nn.ClassNLLCriterion:updateOutput(input, target)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateOutput(input, target)
   end
   if self.weights then
      error('weights not supported (yet!) in clnn.ClassNLLCriterion.  Please an issue on github, to request this functionality')
   end
   if input:dim() ~= 2 then
      error('Input to clnn.ClassNLLCriterion should be 2-d tensor')
   end
   num_samples = input:size(1)
   num_categories = input:size(2)
   if self.buffer == nil then
      self.buffer = input:clone():resize(num_samples,1)
   end
   self.buffer:gather(input, 2, target:unfold(1,1,1))
   self.output = - self.buffer:sum() / num_samples
   return self.output
end


function nn.ClassNLLCriterion:updateGradInput(input, target)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateGradInput(input, target)
   end
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   
   local z = -1
   if self.sizeAverage then
      z = z / target:size(1)
   end
   self.gradInput:scatter(2, target:unfold(1,1,1), z)
   
   return self.gradInput
end

