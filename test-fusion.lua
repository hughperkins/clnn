require 'nngraph'
require 'clnn'

local ngh = require('nodeGraphHelper')
local gh = require('graphHelper')
local fusion = require('fusion')

local fusiontests = {}

function fusiontests.testApplyConvertTanh()
  local x = nn.Identity()()
  local n1 = nn.Tanh()(x)
  fusion.walkConvertToApply(n1)
  tester:asserteq(torch.type(n1.data.module), 'nn.Apply')
  tester:asserteq(torch.type(x.data.module), 'nn.Identity')
  tester:asserteq(n1.data.virtualOutputs, 0)
  tester:asserteq(#n1.data.feobj, 1)
  tester:asserteq(#n1.data.beobj, 1)
  tester:asserteq(n1.data.feobj[1].template, '{{output}} = tanh({{input}});')
  tester:asserteq(n1.data.beobj[1].template, '{{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});')
  tester:asserteq(n1.data.feobj[1].transforms.input, 'input')
  tester:asserteq(n1.data.feobj[1].transforms.output, 'output')
end

function fusiontests.testApplyConvertSigmoid()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  fusion.walkConvertToApply(n1)
  tester:asserteq(torch.type(n1.data.module), 'nn.Apply')
  tester:asserteq(torch.type(x.data.module), 'nn.Identity')
  tester:asserteq(n1.data.virtualOutputs, 0)
  tester:asserteq(#n1.data.feobj, 1)
  tester:asserteq(#n1.data.beobj, 1)
  tester:asserteq(n1.data.feobj[1].template, '{{output}} = 1.f / (1.f + exp( - {{input}}));')
  tester:asserteq(n1.data.beobj[1].template, '{{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});')
  tester:asserteq(n1.data.feobj[1].transforms.input, 'input')
  tester:asserteq(n1.data.feobj[1].transforms.output, 'output')
end

function fusiontests.testApplyConvertTanhSigmoid()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.Tanh()(n1)
  fusion.walkConvertToApply(n2)

  tester:asserteq(torch.type(x.data.module), 'nn.Identity')

  tester:asserteq(torch.type(n2.data.module), 'nn.Apply')
  tester:asserteq(n2.data.virtualOutputs, 0)
  tester:asserteq(#n2.data.feobj, 1)
  tester:asserteq(#n2.data.beobj, 1)
  tester:asserteq(n2.data.feobj[1].template, '{{output}} = tanh({{input}});')
  tester:asserteq(n2.data.beobj[1].template, '{{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});')
  tester:asserteq(n2.data.feobj[1].transforms.input, 'input')
  tester:asserteq(n2.data.feobj[1].transforms.output, 'output')

  tester:asserteq(torch.type(n1.data.module), 'nn.Apply')
  tester:asserteq(n1.data.virtualOutputs, 0)
  tester:asserteq(#n1.data.feobj, 1)
  tester:asserteq(#n1.data.beobj, 1)
  tester:asserteq(n1.data.feobj[1].template, '{{output}} = 1.f / (1.f + exp( - {{input}}));')
  tester:asserteq(n1.data.beobj[1].template, '{{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});')
  tester:asserteq(n1.data.feobj[1].transforms.input, 'input')
  tester:asserteq(n1.data.feobj[1].transforms.output, 'output')
end

function go()
  nloop = n_loop or nloop
  local oldtype = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  -- initSeed(seed)
  tester = torch.Tester()
  tester:add(fusiontests)
  tester:run(tests)
  torch.setdefaulttensortype(oldtype)
  if print_timing then
    print ''
    print ' ------------------------------------------------------------------------------------------------'
    print '|  Module                                                                          |  Speedup    |'
    print ' ------------------------------------------------------------------------------------------------'
    for module,tm in pairs(times) do
      local str = string.format('| %-80s | %4.2f        |', module, (tm.cpu / (tm.gpu or 1e6)))
      print(str)
    end
    print ' ------------------------------------------------------------------------------------------------'
  end
end

go()

