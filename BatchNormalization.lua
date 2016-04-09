require 'nn'

local BN = nn.BatchNormalization

nn.BatchNormalization.baseUpdateOutput = nn.BatchNormalization.updateOutput
nn.BatchNormalization.baseBackward = nn.BatchNormalization.backward
nn.BatchNormalization.baseUpdateGradInput = nn.BatchNormalization.updateGradInput
nn.BatchNormalization.baseAccGradParameters = nn.BatchNormalization.accGradParameters
nn.BatchNormalization.baseAccUpdateGradParameters = nn.BatchNormalization.accUpdateGradParameters

function BN:updateOutput(input)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateOutput(input)
   end

   assert(input:dim() == 2, 'only mini-batch supported (2D tensor), got '
             .. input:dim() .. 'D tensor instead')
   local nBatch = input:size(1)

   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.buffer2 = self.buffer2 or input.new()
   self.centered = self.centered or input.new()
   self.centered:resizeAs(input)
----   self.invstd = self.invstd or input.new()
   self.normalized = self.normalized or input.new()
   self.normalized:resizeAs(input)

   self.output:resizeAs(input)
   self.gradInput:resizeAs(input)

   -- we dont need so many buffers, we can just keep re-using 'buffer', but that 
   -- makes reading the code a bunch harder
   self.mean = self.mean or input.new()
   self.sum = self.sum or input.new()
   self.unbiased_var = self.unbiased_var or input.new()
--   self.mean:resizeAs(self.running_mean)
   self.save_mean = self.save_mean or input.new()
   self.save_mean:resizeAs(self.running_mean)
   self.save_std = self.save_std or input.new()
   self.save_std:resizeAs(self.running_var)

   if self.train == false then
      self.buffer:repeatTensor(self.running_mean, nBatch, 1)
      self.output:add(input, -1, self.buffer)         --  x - E(x)

      self.buffer:add(self.running_var, self.eps):sqrt():cinv()
      self.buffer:repeatTensor(self.buffer, nBatch, 1)
      self.output:cmul(self.buffer)
   else -- training mode
      local n = nBatch
--      local n = input:nElement() / nBatch
      -- calculate mean over mini-batch
      self.mean:mean(input, 1)                        -- E(x) = expectation of x.

      -- self.centered = input - mean(input)
      self.buffer:repeatTensor(self.mean, nBatch, 1)
      self.centered:add(input, -1, self.buffer)         --  x - E(x)

      -- calculate standard deviation over mini-batch
      -- self.buffer = (input - mean(input))^2
      self.sum:resizeAs(self.centered)
      self.sum:copy(self.centered):cmul(self.sum):sum(self.sum, 1) -- [x - E(x)]^2

      -- 1 / E([x - E(x)]^2)
      -- self.save_std = 1 / sqrt[ (input - mean(input))^2 / nBatch + self.eps] )
      self.save_std:div(self.sum, n):add(self.eps):sqrt():pow(-1)

      -- self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean(input)
      self.running_mean:mul(1 - self.momentum):add(self.momentum, self.mean) -- add to running mean

      self.unbiased_var:div(self.sum, n - 1)
      self.running_var:mul(1 - self.momentum):add(self.momentum, self.unbiased_var)

      -- divide standard-deviation + eps
      self.buffer:repeatTensor(self.save_std, nBatch, 1)
      self.output:cmul(self.centered, self.buffer)
      self.normalized:copy(self.output)
   end

   if self.affine then
      -- multiply with gamma and add beta
      self.buffer:repeatTensor(self.weight, nBatch, 1)
      self.output:cmul(self.buffer)
      self.buffer:repeatTensor(self.bias, nBatch, 1)
      self.output:add(self.buffer)
   end

   return self.output
end

function BN:backward(input, gradOutput, scale)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseBackward(input, gradOutput, scale)
   end

   local gradInput = self:updateGradInput(input, gradOutput)
   self:accGradParameters(input, gradOutput, scale)
   return gradInput
end

function BN:updateGradInput(input, gradOutput)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateGradInput(input, gradOutput)
   end

   assert(input:dim() == 2, 'only mini-batch supported')
   assert(gradOutput:dim() == 2, 'only mini-batch supported')
   assert(self.train == true, 'should be in training mode when self.train is true')
   local nBatch = input:size(1)

   self.gradInput:cmul(self.centered, gradOutput)
   self.buffer:mean(self.gradInput, 1)
   self.gradInput:repeatTensor(self.buffer, nBatch, 1)
   self.gradInput:cmul(self.centered):mul(-1)
   self.buffer:repeatTensor(self.save_std, nBatch, 1)
   self.gradInput:cmul(self.buffer):cmul(self.buffer)

   self.buffer:mean(gradOutput, 1)
   self.buffer:repeatTensor(self.buffer, nBatch, 1)
   self.gradInput:add(gradOutput):add(-1, self.buffer)
   self.buffer:repeatTensor(self.save_std, nBatch, 1)
   self.gradInput:cmul(self.buffer)

   if self.affine then
      self.buffer:repeatTensor(self.weight, nBatch, 1)
      self.gradInput:cmul(self.buffer)
   end

   return self.gradInput
end

function BN:accGradParameters(input, gradOutput, scale)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseAccGradParameters(input, gradOutput, scale)
   end

   if self.affine then
      scale = scale or 1.0
      self.buffer2:resizeAs(self.normalized):copy(self.normalized)
      self.buffer2:cmul(gradOutput)
      self.buffer:sum(self.buffer2, 1) -- sum over mini-batch
      self.gradWeight:add(scale, self.buffer)
      self.buffer:sum(gradOutput, 1) -- sum over mini-batch
      self.gradBias:add(scale, self.buffer)
   end
end

