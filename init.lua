require('cltorch')

require "torch"
require 'nn'
--print('nn.Module\n', nn.Module)

require('cltorch')
clnn = paths.require("libclnn")


local function Module__cl(self)
--   print("Module__cl")
 --  print('self\n', self)
  self.weight = self.weight:cl()
  self.bias = self.bias:cl()
  self.output = self.output:cl()
  self.gradInput = self.gradInput:cl()
  self.gradWeight = self.gradWeight:cl()
  self.gradBias = self.gradBias:cl()

--   self.weight 
   return self
end

local function Criterion__cl(self)
--   print("Module__cl")
 --  print('self\n', self)
  self.gradInput = self.gradInput:cl()

  return self
end


--print('nn.Module\n', nn.Module)
rawset(torch.getmetatable('nn.Module'), 'cl', Module__cl)
rawset(torch.getmetatable('nn.Criterion'), 'cl', Criterion__cl)

require 'MSECriterion'

