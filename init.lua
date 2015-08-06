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

if nn.gModule ~= nil then
  function nn.gModule:cl()
  --  print('nn.gModule.cl')
    self:type('torch.ClTensor')
    -- can build shadow graph here...
    return self
  end
end

include 'nodeGraphHelper.lua'
include 'Fusible.lua'
include 'Apply.lua'

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

-- monkey-patch nn.Module __call__
local Module = torch.getmetatable('nn.Module')
Module.__clnn_old_call = Module.__call__
function Module.__call__(self, ...)
	local nArgs = select("#", ...)
	assert(nArgs <= 1, 'Use {input1, input2} to pass multiple inputs.')

	local input = ...
	if nArgs == 1 and input == nil then
		error('what is this in the input? nil')
	end
	if torch.type(input) ~= 'table' then
		input = {input}
	end
  if torch.type(input[1]) ~= 'nn.Fusible' then
    return Module.__clnn_old_call(self, ...)
  end
  -- we've been passed in the inputs nodes, and will create a single child
  -- with these as the inputs
  local child = nn.Fusible({numOutputs=1, numInputs=#input, module=self})
	for i, dnode in ipairs(input) do
		if torch.typename(dnode) ~= 'nn.Fusible' then
			error('what is this in the input? ' .. tostring(dnode))
		end
    dnode:add(child)
	end

	return child
end

