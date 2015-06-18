function torch.ClTensor.nn.Tanh_updateOutput(self, input)
  self.output = torch.tanh(input)
  return self.output
end

function torch.ClTensor.nn.Tanh_updateGradInput(self, input, gradOutput)
  self.gradInput = torch.cmul(gradOutput, - torch.pow(self.output, 2) + 1)
  return self.gradInput
end

