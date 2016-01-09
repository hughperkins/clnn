require 'nn'

nn.LogSoftMax.baseUpdateOutput = nn.LogSoftMax.updateOutput
nn.LogSoftMax.baseUpdateGradInput = nn.LogSoftMax.updateGradInput

function nn.LogSoftMax:updateOutput(input)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateOutput(input)
   end

   if input:dim() == 1 then
      if self.maxbuffer == nil then
         self.maxbuffer, self.resind = input:max(1)
         self.vec_size = input:size(1)
      end
      self.output:resize(input:size())
      
      self.maxbuffer:max(self.resind, input, 1)
      
      self.output:copy(input)
      self.output:csub(self.maxbuffer:expand(input:size(1)))
      self.output:exp()
      self.maxbuffer:sum(self.output,1)
      
      self.output:cdiv(self.maxbuffer:expand(input:size(1)))
      self.output:log()

      return self.output
   elseif input:dim() == 2 then
      if self.maxbuffer == nil then
         self.maxbuffer, self.resind = input:max(2)
         self.vec_size = input:size(2)
      end
      self.output:resize(input:size())
      
      self.maxbuffer:max(self.resind, input, 2)
      
      self.output:copy(input)
      self.output:csub(self.maxbuffer:expand(input:size(1), input:size(2)))
      self.output:exp()
      self.maxbuffer:sum(self.output,2)
      
      self.output:cdiv(self.maxbuffer:expand(input:size(1), input:size(2)))
      self.output:log()

      return self.output
   else
      error('LogSoftMax expects 1-d or 2-d tensor currently')
   end
end

function nn.LogSoftMax:updateGradInput(input, gradOutput)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateGradInput(input, gradOutput)
   end

   local nElement = self.gradInput:nElement()
   self.gradInput:resizeAs(input)
   if self.gradInput:nElement() ~= nElement then
      self.gradInput:zero()
   end

   if input:dim() == 1 then
      self.maxbuffer:sum(gradOutput, 1)
      self.gradInput:copy(self.output)
      self.gradInput:exp()
      self.gradInput:cmul(self.maxbuffer:expand(input:size(1)))
      self.gradInput:neg()
      self.gradInput:add(gradOutput)
   elseif input:dim() == 2 then
      self.maxbuffer:sum(gradOutput, 2)
      self.gradInput:copy(self.output)
      self.gradInput:exp()
      self.gradInput:cmul(self.maxbuffer:expand(input:size(1), input:size(2)))
      self.gradInput:neg()
      self.gradInput:add(gradOutput)
   else
      error('LogSoftMax expects 1-d or 2-d tensor currently')      
   end

   return self.gradInput
end

