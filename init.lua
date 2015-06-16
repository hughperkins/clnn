require "torch"
clnn = paths.require("libclnn")

require('cltorch')

local function Module__cl(self)
--   print("Module__cl")
 --  print('self\n', self)
  self.weight = self.weight:cl()
  self.bias = self.bias:cl()
  self.output = self.output:cl()

--   self.weight 
   return self
end

require 'nn'

rawset(torch.getmetatable('nn.Module'), 'cl', Module__cl)

