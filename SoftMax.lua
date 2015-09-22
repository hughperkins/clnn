function torch.ClTensor.nn.SoftMax_updateOutput(self, input)
   error("Not implemented")

   --  cltorch.setTrace(1)
   if input:dim() ~= 2 then
      error('SoftMax expects 2-d tensor currently')
   end
   if self.maxbuffer == nil then
      --    self.buffer = input:clone()
      self.maxbuffer, self.resind = input:max(2)
      self.vec_size = input:size(2)
   end
   self.output:resize(input:size())
   
   if true then
      if true then
         self.maxbuffer:max(self.resind, input, 2)
         
         self.output:copy(input)
         self.output:csub(self.maxbuffer:expand(input:size(1), input:size(2)))
         self.output:exp()
         self.maxbuffer:sum(self.output,2)
         
         self.output:cdiv(self.maxbuffer:expand(input:size(1), input:size(2)))
         self.output:log()
      end
   end
   --  cltorch.setTrace(0)
   return self.output
end

function torch.ClTensor.nn.SoftMax_updateGradInput(self, input, gradOutput)
   error("Not implemented")

   local nElement = self.gradInput:nElement()
   self.gradInput:resizeAs(input)
   if self.gradInput:nElement() ~= nElement then
      self.gradInput:zero()
   end
   
   --  cltorch.setTrace(1)
   if true then
      self.maxbuffer:sum(gradOutput, 2)
      self.gradInput:copy(self.output)
      self.gradInput:exp()
      self.gradInput:cmul(self.maxbuffer:expand(input:size(1), input:size(2)))
      self.gradInput:neg()
      self.gradInput:add(gradOutput)
   end
   --  cltorch.setTrace(0)
   return self.gradInput
end

