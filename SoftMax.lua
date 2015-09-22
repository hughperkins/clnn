function torch.ClTensor.nn.SoftMax_updateOutput(self, input)
   self.output:resize(input:size())
   if input:dim() == 1 then
      if self.maxbuffer == nil then
         self.maxbuffer, self.resind = input:max(1)
         self.vec_size = input:size(1)
      end

      self.maxbuffer:max(self.resind, input, 1)
      self.output:copy(input)
      self.output:csub(self.maxbuffer:expand(input:size(1)))
      self.output:exp()
      self.maxbuffer:sum(self.output, 1)
      
      self.output:cdiv(self.maxbuffer:expand(input:size(1)))
   elseif input:dim() == 2 then
      if self.maxbuffer == nil then
         self.maxbuffer, self.resind = input:max(2)
         self.vec_size = input:size(2)
      end

      self.maxbuffer:max(self.resind, input, 2)
      
      self.output:copy(input)
      self.output:csub(self.maxbuffer:expand(input:size(1), input:size(2)))
      self.output:exp()
      self.maxbuffer:sum(self.output,2)
      
      self.output:cdiv(self.maxbuffer:expand(input:size(1), input:size(2)))
   else
      error('SoftMax expects 1-d or 2-d tensor currently')
   end

   return self.output
end

function torch.ClTensor.nn.SoftMax_updateGradInput(self, input, gradOutput)
   local nElement = self.gradInput:nElement()
   self.gradInput:resizeAs(input)
   if self.gradInput:nElement() ~= nElement then
      self.gradInput:zero()
   end
   if self._gradBuffer == nil then
      self._gradBuffer = gradOutput:clone()
   end
   self._gradBuffer:resize(gradOutput:size())
   if input:dim() == 1 then
      self._gradBuffer:copy(self.output)
      self._gradBuffer:cmul(gradOutput)
      self.maxbuffer:sum(self._gradBuffer, 1)
      self.gradInput:copy(gradOutput)
      self.gradInput:csub(self.maxbuffer:expand(input:size(1)))
      self.gradInput:cmul(self.output)
   elseif input:dim() == 2 then
      self._gradBuffer:copy(self.output)
      self._gradBuffer:cmul(gradOutput)
      self.maxbuffer:sum(self._gradBuffer, 2)
      self.gradInput:copy(gradOutput)
      self.gradInput:csub(self.maxbuffer:expand(input:size(1), input:size(2)))
      self.gradInput:cmul(self.output)
   else
      error('SoftMax expects 1-d or 2-d tensor currently')
   end
   return self.gradInput
end

