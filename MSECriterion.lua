function torch.ClTensor.nn.MSECriterion_updateOutput(self, input, target)
   -- assume 2 dim for now, why would anyone use 1 dim really?
   assert (input:dim() == 2)
   local nframe = input:size(1)
   --    self.output:resize(nframe, self.bias:size(1))
   --    if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
   --       print('allocate addbuffer')
   --       self.addBuffer = input.new(nframe):fill(1)
   --    end
   if self.workBuffer == nil then
      print('allocate workbuffer')
      self.workBuffer = input:clone()
   end
   
   se = 0
   
   self.workBuffer:copy(input)
   self.workBuffer:csub(target)
   self.workBuffer:pow(2)
   se = torch.sum(self.workBuffer)
   
   --  work = input - target
   --  work = torch.pow(work, 2)
   --  se = torch.sum(work)
   
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
   --  self.gradInput = (input - target) * norm
   return self.gradInput
end

