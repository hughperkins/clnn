require 'nn'

nn.MSECriterion.baseUpdateOutput = nn.MSECriterion.updateOutput
nn.MSECriterion.baseUpdateGradInput = nn.MSECriterion.updateGradInput

function nn.MSECriterion:updateOutput(input, target)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateOutput(input, target)
   end

   if self.workBuffer == nil then
      self.workBuffer = input:clone()
   end

   self.workBuffer:resizeAs(input)
   self.workBuffer:map2_on_gpu(input, target, "*out = (*in1 - *in2) * (*in1 - *in2)")
   local se = torch.sum(self.workBuffer)
   
   local mse = se
   if self.sizeAverage then
      mse = se / torch.numel(input)
   end
   return mse
end

function nn.MSECriterion:updateGradInput(input, target)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateGradInput(input, target)
   end

   local norm = 2
   if self.sizeAverage then
      local size = torch.numel(input)
      norm = norm / size
   end
   self.gradInput:resize(target:size())
   self.gradInput:map2_on_gpu(input, target, "*out = " .. norm .. " * (*in1 - *in2)")
   return self.gradInput
end

