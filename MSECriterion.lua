function torch.ClTensor.nn.MSECriterion_updateOutput(self, input, target)
  work = input - target
  work = torch.pow(work, 2)
  se = torch.sum(work)
  if self.sizeAverage then
    mse = se / torch.numel(input)
  end
  return mse
end

function torch.ClTensor.nn.MSECriterion_updateGradInput(self, input, target)
  norm = 2
  if self.sizeAverage then
    size = torch.numel(input)
    norm = norm / size
  end
  self.gradInput = (input - target) * norm
  return self.gradInput
end

