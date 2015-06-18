function torch.ClTensor.nn.MSECriterion_updateGradInput(self, input, target)
--  print('torch.ClTensor.nn.MSECriterion_updateGradInput')
--  print('self', self)
--  print('input', input)
--  print('target', target)
  norm = 2
  if self.sizeAverage then
    size = torch.numel(input)
    norm = norm / size
  end
  self.gradInput = (input - target) * norm
  return self.gradInput
end

function torch.ClTensor.nn.MSECriterion_updateOutput(self, input, target)
--  print('torch.ClTensor.nn.MSECriterion_updateOutput')
--  print('input\n', input)
--  print('target\n', target)
--  print('input size', input:size())
--  print('target size', target:size())
  work = input - target
  work = torch.pow(work, 2)
--  print('diffsquare\n', work)
  se = torch.sum(work)
--  print('se\n', se)
  if self.sizeAverage then
    mse = se / torch.numel(input)
  end
--  print('mse\n', mse)
  return mse
end



