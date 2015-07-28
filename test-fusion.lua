require 'nngraph'
require 'clnn'

local ngh = require('nodeGraphHelper')
local gh = require('graphHelper')
local fusion = require('fusion')

local fusiontests = {}

function normalizeWhite(input)
  old = nil
  while old ~= input do
    old = input
    input = input:gsub('  ', ' ')
  end
  while old ~= input do
    old = input
    input = input:gsub(' \n', '\n')
  end
  while old ~= input do
    old = input
    input = input:gsub('\n\n', '\n')
  end
  return input
end

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

function fusiontests.testFuseTanhSigmoid()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.Tanh()(n1)

  ngh.nodeSetName(x, 'x')
  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')

  ngh.walkAddParents(n2)
  ngh.walkStripByObjects(n2)
  ngh.walkReverseAddDataIds(x )

  fusion.reverseWalkConvertToApply(x)
  tester:asserteq(ngh.count(n2), 3)
  fusion.doFuse(x)
  tester:asserteq(ngh.count(n2), 2)

  tester:asserteq(torch.type(x.data.module), 'nn.Identity')

--  tester:asserteq(torch.type(n2.data.module), 'nn.Apply')
--  tester:asserteq(n2.data.virtualOutputs, 0)
--  tester:asserteq(#n2.data.feobj, 1)
--  tester:asserteq(#n2.data.beobj, 1)
--  tester:asserteq(n2.data.feobj[1].template, '{{output}} = tanh({{input}});')
--  tester:asserteq(n2.data.beobj[1].template, '{{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});')
--  tester:asserteq(n2.data.feobj[1].transforms.input, 'input')
--  tester:asserteq(n2.data.feobj[1].transforms.output, 'output')

--  tester:asserteq(torch.type(n1.data.module), 'nn.Apply')
--  tester:asserteq(n1.data.virtualOutputs, 0)
--  tester:asserteq(#n1.data.feobj, 1)
--  tester:asserteq(#n1.data.beobj, 1)
--  tester:asserteq(n1.data.feobj[1].template, '{{output}} = 1.f / (1.f + exp( - {{input}}));')
--  tester:asserteq(n1.data.beobj[1].template, '{{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});')
--  tester:asserteq(n1.data.feobj[1].transforms.input, 'input')
--  tester:asserteq(n1.data.feobj[1].transforms.output, 'output')

  tester:asserteq(ngh.nodeGetName(x), 'x')
  tester:asserteq(#x.parents, 1)
  tester:asserteq(#x.parents[1].parents, 0)
  local fused = x.parents[1]
  local fdat = fused.data
  tester:asserteq(ngh.nodeGetName(fused), 'n1.n2')
  tester:asserteq(#fdat.feobj, 2)
  tester:asserteq(fdat.feobj[1].template, '{{output}} = 1.f / (1.f + exp( - {{input}}));')
  tester:asserteq(fdat.feobj[2].template, '{{output}} = tanh({{input}});')
  tester:asserteq(fdat.feobj[1].transforms.input, 'input')
  tester:asserteq(fdat.feobj[1].transforms.output, 'float virtualOutput1')
  tester:asserteq(fdat.feobj[2].transforms.input, 'virtualOutput1')
  tester:asserteq(fdat.feobj[2].transforms.output, 'output')
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

