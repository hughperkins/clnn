function torch.ClTensor.nn.MSECriterion_updateOutput(self, input, target)
   if self.workBuffer == nil then
      self.workBuffer = input:clone()
   end
   
   se = 0   
   self.workBuffer:copy(input)
   self.workBuffer:csub(target)
   self.workBuffer:pow(2)
   se = torch.sum(self.workBuffer)
   
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
   self.gradInput:resize(target:size())
   self.gradInput:map2(input, target, "*out = " .. norm .. " * (*in1 - *in2)")
   return self.gradInput
end

