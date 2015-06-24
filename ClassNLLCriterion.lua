require 'nn'

-- monkey patch it in :-P

nn.ClassNLLCriterion.baseUpdateOutput = nn.ClassNLLCriterion.updateOutput

print('nn.ClassNLLCriterion', nn.ClassNLLCriterion)
print('nn.ClassNLLCriterion.updateOutput', nn.ClassNLLCriterion.updateOutput)

-- reminder to self: we assume 2-dimensional input
-- dimension 1 is the sampel number, and is the row number
-- dimension 2 is the value for each of the output neurons for that sample
-- we're going to reduce dimension 2 to a size of 1, for each sample,
-- using the knowledge of which is the ground truth class for taht
-- sample.  ie, the loss for that sample is minus the input value
-- for whichever neuron matches the ground truth class
function nn.ClassNLLCriterion:updateOutput(input, target)
  print('monkey patch classnllcriterion update output')
  print(torch.type(input))
  if torch.type(input) ~= 'torch.ClTensor' then
    return self:baseUpdateOutput(input, target)
  end
  print('cltensor classnllcriterion updateoutput')
  print('input\n', input)
  print('target\n', target)
  if self.weights then
    error('weights not supported (yet!) in clnn.ClassNLLCriterion.  Please an issue on github, to request this functionality')
  end
  num_samples = input:size(1)
  num_categories = input:size(2)
  print('N', num_samples, 'categories', num_categories)
  if self.buffer == nil then
    self.buffer = input:clone():resize(num_samples,1)
  end
  print('self.buffer', self.buffer)
  self.buffer:gather(input, 2, target:unfold(1,1,1))
  print('self.buffer', self.buffer)
  self.output = - self.buffer:sum() / num_samples
  return self.output
end

