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

-- next few lines should be in MSECriterion.lua really, but putting 
-- them here for now, "to get it working"
function torch.ClTensor.nn.MSECriterion_updateGradInput(self, input, target)
--  print('torch.ClTensor.nn.MSECriterion_updateGradInput')
--  print('self', self)
--  print('input', input)
--  print('target', target)
  norm = 2
  if self.sizeAverage then
    size = torch.numel(input)
    norm = norm / size
  end
  self.gradInput = (input - target) * norm
  return self.gradInput
end

function torch.ClTensor.nn.MSECriterion_updateOutput(self, input, target)
--  print('torch.ClTensor.nn.MSECriterion_updateOutput')
  work = input - target
  work = torch.pow(work, 2)
--  print('diffsquare\n', work)
  se = torch.sum(work)
--  print('se\n', se)
  if self.sizeAverage then
    mse = se / torch.numel(input)
  end
--  print('mse\n', mse)
  return mse
end


