require('cltorch')

require "torch"
require 'nn'
--print('nn.Module\n', nn.Module)

-- require('cltorch')

--torch.ClTensor.nn = {}

clnn = paths.require("libclnn")


function nn.Module:cl()
   return self:type('torch.ClTensor')
end

function nn.Criterion:cl()
   return self:type('torch.ClTensor')
end

include 'Tanh.lua'
include 'Sigmoid.lua'
include 'Threshold.lua'
include 'LogSoftMax.lua'
include 'Narrow.lua'

include 'MSECriterion.lua'
include 'ClassNLLCriterion.lua'

include 'StatefulTimer.lua'

