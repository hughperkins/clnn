require 'cltorch'

require "torch"
require 'nn'
require('clnn.THCLNN')

clnn = paths.require("libclnn")


function nn.Module:cl()
  return self:type('torch.ClTensor')
end

function nn.Criterion:cl()
  return self:type('torch.ClTensor')
end

--include 'SpatialConvolutionMM.lua'
include 'SpatialAveragePooling.lua'
include 'SpatialMaxPooling.lua'
include 'ELU.lua'
include 'Tanh.lua'
include 'Pointwise.lua'
include 'Threshold.lua'
include 'LogSoftMax.lua'
-- include 'SoftMax.lua'
include 'Narrow.lua'

include 'MSECriterion.lua'
include 'ClassNLLCriterion.lua'

include 'StatefulTimer.lua'
include 'CMulTable.lua'

include 'test.lua'

