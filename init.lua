require('cltorch')

require "torch"
require 'nn'

clnn = paths.require("libclnn")

function nn.Module:cl()
   return self:type('torch.ClTensor')
end

function nn.Criterion:cl()
   return self:type('torch.ClTensor')
end

function nn.gModule:cl()
--  print('nn.gModule.cl')
  self:type('torch.ClTensor')
  -- can build shadow graph here...
  return self
end

include 'Apply.lua'
include 'gApply.lua'

include 'Tanh.lua'
include 'Pointwise.lua'
include 'Threshold.lua'
include 'LogSoftMax.lua'
include 'Narrow.lua'

include 'MSECriterion.lua'
include 'ClassNLLCriterion.lua'

include 'StatefulTimer.lua'
include 'CMulTable.lua'

include 'test.lua'

