require('cltorch')

require "torch"
require 'nn'
--print('nn.Module\n', nn.Module)

require('cltorch')

--torch.ClTensor.nn = {}

clnn = paths.require("libclnn")


function nn.Module:cl()
   return self:type('torch.ClTensor')
end

function nn.Criterion:cl()
   return self:type('torch.ClTensor')
end

require 'Tanh'
require 'Sigmoid'
require 'FullyConnected'
require 'Threshold'
require 'LogSoftMax'

require 'MSECriterion'
require 'ClassNLLCriterion'

