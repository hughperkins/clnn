require('cltorch')

require "torch"
require 'nn'
--print('nn.Module\n', nn.Module)

require('cltorch')
clnn = paths.require("libclnn")


function nn.Module:cl()
   return self:type('torch.ClTensor')
end

function nn.Criterion:cl()
   return self:type('torch.ClTensor')
end

require 'MSECriterion'
require 'Tanh'
require 'Sigmoid'

