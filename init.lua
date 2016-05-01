-- check we are installed from distro, otherwise error message and exit...

require 'os'

xpcall(function()
  require 'distrocheck'
end, function()
  print('Please install cltorch from distro, per instructions at https://github.com/hughperkins/cltorch')
  os.exit(1)
end)

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

torch.ClTensor.nn = {}

include 'LookupTable.lua'
include 'Pointwise.lua'
include 'Threshold.lua'
include 'LogSoftMax.lua'
include 'Narrow.lua'

include 'MSECriterion.lua'
include 'ClassNLLCriterion.lua'

include 'StatefulTimer.lua'
include 'CMulTable.lua'

include 'test.lua'

--include 'SpatialUpSamplingNearest.lua'
