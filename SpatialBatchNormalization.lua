local BN = nn.SpatialBatchNormalization

nn.SpatialBatchNormalization.baseUpdateOutput2 = nn.SpatialBatchNormalization.updateOutput
nn.SpatialBatchNormalization.baseBackward2 = nn.SpatialBatchNormalization.backward
nn.SpatialBatchNormalization.baseUpdateGradInput2 = nn.SpatialBatchNormalization.updateGradInput
nn.SpatialBatchNormalization.baseAccGradParameters2 = nn.SpatialBatchNormalization.accGradParameters
nn.SpatialBatchNormalization.baseAccUpdateGradParameters2 = nn.SpatialBatchNormalization.accUpdateGradParameters

function BN:updateOutput(input)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateOutput2(input)
   end

   assert(input:dim() == 4, 'only mini-batch supported (4D tensor), got '
             .. input:dim() .. 'D tensor instead')
   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)
   local nInput = input:size(2)
   local n = input:nElement() / nInput

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

   self.output:resizeAs(input)
   if self.train == false then
      self.buffer:repeatTensor(self.running_mean:view(1, nFeature, 1, 1), nBatch, 1, iH, iW)
      self.output:add(input, -1, self.buffer)         --  x - E(x)

      self.buffer:add(self.running_var, self.eps):sqrt():cinv()
      self.buffer:repeatTensor(self.buffer:view(1, nFeature, 1, 1), nBatch, 1, iH, iW)
      self.output:cmul(self.buffer)
   else -- training mode
      -- calculate mean over mini-batch
      local in_folded = input:view(nBatch, nFeature, iH * iW)
--      print('nInput', nInput, 'n', n)
--      print('input', input)
--      print('in_folded', in_folded)
      self.buffer:sum(in_folded, 1)
--      print('self.buffer', self.buffer)
      self.mean:sum(self.buffer, 3)
--      print('self.mean', self.mean)
      self.mean:div(n)                        -- E(x) = expectation of x.
--      self.mean:mean(self.mean, 3)
--      print('self.mean', self.mean)

      -- self.centered = input - mean(input)
--      print('input:size()', input:size())
--      print('self.mean:size()', self.mean:size())
      self.buffer:repeatTensor(self.mean:view(1, nFeature, 1, 1), nBatch, 1, iH, iW)
--      print('self.buffer:size()', self.buffer:size())
      self.centered:add(input, -1, self.buffer)         --  x - E(x)

      -- calculate standard deviation over mini-batch
      -- self.buffer = (input - mean(input))^2
      self.sum:resizeAs(self.centered)
      self.sum:copy(self.centered)
--      self.sum = self.sum:view(nBatch, nFeature, iH * iW)
      self.sum:cmul(self.sum)
      self.buffer:sum(self.sum, 1)
      self.buffer2:sum(self.buffer, 3)
      self.sum:sum(self.buffer2, 4) -- [x - E(x)]^2

      -- 1 / E([x - E(x)]^2)
      -- self.save_std = 1 / sqrt[ (input - mean(input))^2 / nBatch + self.eps] )
      self.save_std:div(self.sum, n):add(self.eps):sqrt():pow(-1)

      -- self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean(input)
--      print('self.running_mean:size()', self.running_mean:size())
      self.running_mean:mul(1 - self.momentum):add(self.momentum, self.mean) -- add to running mean

--      print('self.sum:size()', self.sum:size())
--      print('self.unbiased_var:size()', self.unbiased_var:size())
      self.unbiased_var:div(self.sum, n - 1)
      self.running_var:mul(1 - self.momentum):add(self.momentum, self.unbiased_var)

      -- divide standard-deviation + eps
      self.buffer:repeatTensor(self.save_std, nBatch, 1, iH, iW)
      self.output:cmul(self.centered, self.buffer)
      self.normalized:copy(self.output)
   end

   if self.affine then
      -- multiply with gamma and add beta
      self.buffer:repeatTensor(self.weight:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
      self.output:cmul(self.buffer)
      self.buffer:repeatTensor(self.bias:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
      self.output:add(self.buffer)
   end

   return self.output
end

function BN:backward(input, gradOutput, scale)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseBackward2(input, gradOutput, scale)
   end

   local gradInput = self:updateGradInput(input, gradOutput)
   self:accGradParameters(input, gradOutput, scale)
   return gradInput
end

function BN:updateGradInput(input, gradOutput)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateGradInput2(input, gradOutput)
   end

   assert(input:dim() == 4, 'only mini-batch supported')
   assert(gradOutput:dim() == 4, 'only mini-batch supported')
   assert(self.train == true, 'should be in training mode when self.train is true')
   local nBatch = input:size(1)
   local nFeature = input:size(2)
   local iH = input:size(3)
   local iW = input:size(4)

   self.gradInput:cmul(self.centered, gradOutput)
   local gi_folded = self.gradInput:view(nBatch, nFeature, iH * iW)
   self.buffer2:mean(self.buffer:mean(gi_folded, 1), 3)
   self.gradInput:repeatTensor(self.buffer2:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
   self.gradInput:cmul(self.centered):mul(-1)
   self.buffer:repeatTensor(self.save_std:view(1, nFeature, 1, 1),
                            nBatch, 1, iH, iW)
   self.gradInput:cmul(self.buffer):cmul(self.buffer)

   self.buffer:mean(gradOutput:view(nBatch, nFeature, iH*iW), 1)
   self.buffer2:mean(self.buffer, 3)
   self.buffer:repeatTensor(self.buffer2:view(1, nFeature, 1, 1),
                            nBatch, 1, iH, iW)
   self.gradInput:add(gradOutput):add(-1, self.buffer)
   self.buffer:repeatTensor(self.save_std:view(1, nFeature, 1, 1),
                            nBatch, 1, iH, iW)
   self.gradInput:cmul(self.buffer)

   if self.affine then
      self.buffer:repeatTensor(self.weight:view(1, nFeature, 1, 1),
                               nBatch, 1, iH, iW)
      self.gradInput:cmul(self.buffer)
   end

   return self.gradInput
end

function BN:accGradParameters(input, gradOutput, scale)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseAccGradParameters2(input, gradOutput, scale)
   end

   if self.affine then
      scale = scale or 1.0
      local nBatch = input:size(1)
      local nFeature = input:size(2)
      local iH = input:size(3)
      local iW = input:size(4)
      self.buffer2:resizeAs(self.normalized):copy(self.normalized)
      self.buffer2 = self.buffer2:cmul(gradOutput):view(nBatch, nFeature, iH*iW)
      self.buffer:sum(self.buffer2, 1) -- sum over mini-batch
      self.buffer2:sum(self.buffer, 3) -- sum over pixels
      self.gradWeight:add(scale, self.buffer2)

      self.buffer:sum(gradOutput:view(nBatch, nFeature, iH*iW), 1)
      self.buffer2:sum(self.buffer, 3)
      self.gradBias:add(scale, self.buffer2) -- sum over mini-batch
   end
end

