require 'nn'

nn.LookupTable.baseUpdateOutput = nn.LookupTable.updateOutput
--nn.LookupTable.baseUpdateGradInput = nn.LookupTable.updateGradInput
nn.LookupTable.baseAccGradParameters = nn.LookupTable.accGradParameters
nn.LookupTable.baseAccUpdateGradParameters = nn.LookupTable.accUpdateGradParameters

function nn.LookupTable:__init(nIndex, nOutput)
   nn.Module.__init(self)

   -- self.nIndex = nIndex
   -- self.nOutput = nOutput
   self.weight = torch.Tensor(nIndex, nOutput)
   self.gradWeight = torch.Tensor(nIndex, nOutput):zero()

   self:reset()
end

--function LookupTable:updateOutput(input)
--   if torch.type(input) ~= 'torch.ClTensor' then
--      return self:baseUpdateOutput(input)
--   end

--   self:backCompatibility()
--   input = self:makeInputContiguous(input)
--   if input:dim() == 1 then
--      self.output:index(self.weight, 1, input)
--   elseif input:dim() == 2 then
--      self.output:index(self.weight, 1, input:view(-1))
--      self.output = self.output:view(input:size(1), input:size(2), self.weight:size(2))
--   else
--      error("input must be a vector or matrix")
--   end
--   return self.output
--end

--function LookupTable:accGradParameters(input, gradOutput, scale)
--   if torch.type(input) ~= 'torch.ClTensor' then
--      return self:baseAccGradParameters(input)
--   end

--   self:backCompatibility()
--   input = self.copiedInput and self._input or input
--   if input:dim() == 2 then
--      input = input:view(-1)
--   elseif input:dim() ~= 1 then
--      error("input must be a vector or matrix")
--   end
--   self.gradWeight.nn.LookupTable_accGradParameters(self, input, gradOutput, scale)
--end

--local LookupTable, parent = torch.class('nn.LookupTable', 'nn.Module')

--LookupTable.__version = 3

--function LookupTable:__init(nIndex, ...)
--   parent.__init(self)
--   local arg = {...}

--   if select('#', ...) == 1 and type(arg[1]) ~= "number" then
--      local size = arg[1]
--      self.size = torch.LongStorage(#size + 1)
--      for i=1,#size do
--         self.size[i+1] = size[i]
--      end
--   else
--      self.size = torch.LongStorage(select('#', ...)+1)
--      for i=1,select('#',...) do
--         self.size[i+1] = arg[i]
--      end
--   end

--   self.size[1] = nIndex
--   
--   batchSize = torch.LongTensor(#self.size + 1)
--   batchSize:narrow(1, 2,#self.size):copy(torch.LongTensor(self.size))
--   batchSize[1] = 1
--   self.batchSize = batchSize:storage()
--   
--   self.weight = torch.Tensor(self.size)
--   self.gradWeight = torch.Tensor(self.size):zero()
--   self.inputs = {}

--   self.nBackward = 0
--   self:reset()
--end

--function LookupTable:reset(stdv)
--   stdv = stdv or 1
--   if nn.oldSeed then
--      self.weight:apply(function()
--         return torch.normal(0, stdv)
--      end)
--   else
--      self.weight:normal(0, stdv)
--   end
--end

function nn.LookupTable:updateOutput(input)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseUpdateOutput(input)
   end

   assert(not self.shouldScaleGradByFreq, 'self.shouldScaleGradByFreq not implemented')

   if self.size == nil then
      self.size = self.weight:size()

--     if select('#', ...) == 1 and type(arg[1]) ~= "number" then
--        local size = arg[1]
--        self.size = torch.LongStorage(#size + 1)
--        for i=1,#size do
--           self.size[i+1] = size[i]
--        end
--     else
--        self.size = torch.LongStorage(select('#', ...)+1)
--        for i=1,select('#',...) do
--           self.size[i+1] = arg[i]
--        end
--     end

--     self.size[1] = nIndex
     
      batchSize = torch.LongTensor(#self.size + 1)
      batchSize:narrow(1, 2,#self.size):copy(torch.LongTensor(self.size))
      batchSize[1] = 1
      self.batchSize = batchSize:storage()

      self.nBackward = 0
      self.inputs = {}
   end

   if input:dim() == 1 then
      local nIndex = input:size(1)
      self.size[1] = nIndex
      self.output:resize(self.size)
      for i=1,nIndex do
         self.output:select(1, i):copy(self.weight:select(1, input[i]))
      end
   elseif input:dim() == 2 then
      local nExample = input:size(1)
      local nIndex = input:size(2)
      self.batchSize[1] = nExample
      self.batchSize[2] = nIndex
      self.output:resize(self.batchSize)
      
      for i=1,nExample do
         local output = self.output:select(1, i)
         local input = input:select(1, i)
         for j=1,nIndex do
            output:select(1, j):copy(self.weight:select(1, input[j]))
         end
      end
   end

   return self.output
end

--function LookupTable:zeroGradParameters()
--   for k,_ in pairs(self.inputs) do
--      self.gradWeight:select(1, k):zero()
--   end
--   self.inputs = {}
--   self.nBackward = 0
--end

function nn.LookupTable:accGradParameters(input, gradOutput, scale)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseAccGradParameters(input, gradOutput, scale)
   end

   assert(not self.shouldScaleGradByFreq, 'self.shouldScaleGradByFreq not implemented')

   scale = scale or 1
   if input:dim() == 1 then
      self.nBackward = self.nBackward + 1
      for i=1,input:size(1) do
         local k = input[i]
         self.inputs[k] = (self.inputs[k] or 0) + 1
         self.gradWeight:select(1, k):add(scale, gradOutput:select(1, i))
      end
   elseif input:dim() == 2 then
      self.nBackward = self.nBackward + input:size(1)
      for i=1,input:size(1) do
         local input = input:select(1, i)
         local gradOutput = gradOutput:select(1, i)
         for j=1,input:size(1) do
            local k = input[j]
            self.inputs[k] = (self.inputs[k] or 0) + 1
            self.gradWeight:select(1, k):add(scale, gradOutput:select(1, j))
         end
      end
   end
end

function nn.LookupTable:accUpdateGradParameters(input, gradOutput, lr)
   if torch.type(input) ~= 'torch.ClTensor' then
      return self:baseAccUpdateGradParameters(input, gradOutput, lr)
   end

   assert(not self.shouldScaleGradByFreq, 'self.shouldScaleGradByFreq not implemented')

   if input:dim() == 1 then
      for i=1,input:size(1) do
         local k = input[j]
         local kscale = self:scaleUpdateByKey(k)
         self.weight:select(1, input[i]):add(-lr*kscale, gradOutput:select(1, i))
      end
   elseif input:dim() == 2 then 
      for i=1,input:size(1) do
         local input = input:select(1, i)
         local gradOutput = gradOutput:select(1, i)
         for j=1,input:size(1) do
            local k = input[j]
            local kscale = self:scaleUpdateByKey(k)
            self.weight:select(1, k):add(-lr*kscale, gradOutput:select(1, j))
         end
      end
   end
end

--function LookupTable:updateParameters(learningRate)
--   for k,nBackward in pairs(self.inputs) do
--      local kscale = self:scaleUpdateByKey(k)
--      self.weight:select(1, k):add(-learningRate*kscale, self.gradWeight:select(1, k))
--   end
--end

---- scale the update for each key
--function LookupTable:scaleUpdateByKey(inputKey)
--   -- default is to perform no key-based scalling
--   return 1
--end

---- we do not need to accumulate parameters when sharing
--LookupTable.sharedAccUpdateGradParameters = LookupTable.accUpdateGradParameters
