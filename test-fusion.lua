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

  ngh.walkAddParents(n1)
  x = ngh.invertGraph(n1)
  ngh.walkRemoveBidirectional(x)

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

  ngh.walkAddParents(n1)
  x = ngh.invertGraph(n1)
  ngh.walkRemoveBidirectional(x)
  tester:asserteq(ngh.walkValidate(x), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.walkValidate(x), true)

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

  ngh.walkAddParents(n2)
  x = ngh.invertGraph(n2)
  ngh.walkRemoveBidirectional(x)
  tester:asserteq(ngh.walkValidate(x), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.walkValidate(x), true)

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
  ngh.walkRemoveBidirectional(n2)
  tester:asserteq(ngh.walkValidate(n2), true)
  x = ngh.invertGraph(n2)
  ngh.walkAddDataIds(x)
  tester:asserteq(ngh.walkValidate(x), true)

  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.count(x), 3)
  tester:asserteq(ngh.walkValidate(x), true)
  fusion.doFuse(x)
  tester:asserteq(ngh.walkValidate(x), true)
  tester:asserteq(ngh.count(x), 2)

  tester:asserteq(torch.type(x.data.module), 'nn.Identity')

  tester:asserteq(ngh.nodeGetName(x), 'x')
  tester:asserteq(#x.children, 1)
  tester:asserteq(#x.children[1].children, 0)
  local fused = x.children[1]
  local fdat = fused.data
  tester:asserteq(ngh.nodeGetName(fused), 'n2.n1')
  tester:asserteq(#fdat.feobj, 2)
  tester:asserteq(fdat.feobj[1].template, '{{output}} = 1.f / (1.f + exp( - {{input}}));')
  tester:asserteq(fdat.feobj[2].template, '{{output}} = tanh({{input}});')
  tester:asserteq(fdat.feobj[1].transforms.input, 'input')
  tester:asserteq(fdat.feobj[1].transforms.output, 'float virtualOutput1')
  tester:asserteq(fdat.feobj[2].transforms.input, 'virtualOutput1')
  tester:asserteq(fdat.feobj[2].transforms.output, 'output')
end

function fusiontests.testFuseExpTanhSigmoid()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.Tanh()(n1)
  local n3 = nn.Exp()(n2)

  ngh.nodeSetName(x, 'x')
  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')
  ngh.nodeSetName(n3, 'n3')

  ngh.walkAddParents(n3)
  ngh.walkRemoveBidirectional(n3)
  x = ngh.invertGraph(n2)
  ngh.walkAddDataIds(x)

  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.count(x), 4)
  fusion.doFuse(x)
  tester:asserteq(ngh.walkValidate(x), true)
  tester:asserteq(ngh.count(x), 2)

  tester:asserteq(torch.type(x.data.module), 'nn.Identity')

  tester:asserteq(ngh.nodeGetName(x), 'x')
  tester:asserteq(#x.children, 1)
  tester:asserteq(#x.children[1].children, 0)

  local fused = x.children[1]
  local fdat = fused.data
  tester:asserteq(ngh.nodeGetName(fused), 'n3.n2.n1')
  tester:asserteq(#fdat.feobj, 3)
  tester:asserteq(fdat.feobj[1].template, '{{output}} = 1.f / (1.f + exp( - {{input}}));')
  tester:asserteq(fdat.feobj[2].template, '{{output}} = tanh({{input}});')
  tester:asserteq(fdat.feobj[3].template, '{{output}} = exp({{input}});')

  tester:asserteq(fdat.feobj[1].transforms.input, 'input')
  tester:asserteq(fdat.feobj[1].transforms.output, 'float virtualOutput1')

  tester:asserteq(fdat.feobj[2].transforms.input, 'virtualOutput1')
  tester:asserteq(fdat.feobj[2].transforms.output, 'float virtualOutput2')

  tester:asserteq(fdat.feobj[3].transforms.input, 'virtualOutput2')
  tester:asserteq(fdat.feobj[3].transforms.output, 'output')
end

function fusiontests.testApplyConvertSigmoidAddTable()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.CAddTable()({n1, x})

  ngh.nodeSetName(x, 'x')
  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')

  ngh.walkAddParents(n2)
  ngh.walkRemoveBidirectional(n2)
  x = ngh.invertGraph(n2)
  ngh.walkAddDataIds(x)

  fusion.walkConvertToApply(x)
--  ngh.dot(x, '', 'add')
  tester:asserteq(ngh.count(x), 3)
  print('')
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.dot(x, '', 'xold')
  tester:asserteq(ngh.walkValidate(x), true)
  local xold = ngh.walkClone(x)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraph(x)
  fusion.doFuse(x)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraph(x)
  ngh.dot(x, '', 'xnew')
  tester:asserteq(ngh.count(x), 2)

  tester:asserteq(torch.type(x.data.module), 'nn.Identity')

  local fused = x.children[1]
  local fdat = fused.data
  tester:asserteq(ngh.nodeGetName(fused), 'n2.n1')
  tester:asserteq(#fdat.feobj, 2)
  tester:asserteq(fdat.feobj[1].template, '{{output}} = 1.f / (1.f + exp( - {{input}}));')
  tester:asserteq(fdat.feobj[2].template, '{{output}} = {{input1}} + {{input2}};')

  for k, v in pairs(fdat.feobj[1].transforms) do
    print('feobj[1]', k, v)
  end
  for k, v in pairs(fdat.feobj[2].transforms) do
    print('feobj[2]', k, v)
  end

  tester:asserteq(fdat.feobj[1].transforms.input, 'input')
  tester:asserteq(fdat.feobj[1].transforms.output, 'float virtualOutput1')

  tester:asserteq(fdat.feobj[2].transforms.input1, 'virtualOutput1')
  tester:asserteq(fdat.feobj[2].transforms.input2, 'input2')
  tester:asserteq(fdat.feobj[2].transforms.output, 'output')
end

function fusiontests.testApplyConvertSigmoidAddTable2()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.CAddTable()({x, n1})

  ngh.nodeSetName(x, 'x')
  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')

  ngh.walkAddParents(n2)
  ngh.walkRemoveBidirectional(n2)
  x = ngh.invertGraph(n2)
  ngh.walkAddDataIds(x)

  fusion.walkConvertToApply(x)
--  ngh.dot(x, '', 'add')
  tester:asserteq(ngh.count(x), 3)
  print('')
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.dot(x, '', 'xold')
  tester:asserteq(ngh.walkValidate(x), true)
  local xold = ngh.walkClone(x)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraph(x)
  fusion.doFuse(x)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraph(x)
  ngh.dot(x, '', 'xnew')
  tester:asserteq(ngh.count(x), 2)

  tester:asserteq(torch.type(x.data.module), 'nn.Identity')

  local fused = x.children[1]
  local fdat = fused.data
  tester:asserteq(ngh.nodeGetName(fused), 'n2.n1')
  tester:asserteq(#fdat.feobj, 2)
  tester:asserteq(fdat.feobj[1].template, '{{output}} = 1.f / (1.f + exp( - {{input}}));')
  tester:asserteq(fdat.feobj[2].template, '{{output}} = {{input1}} + {{input2}};')

  for k, v in pairs(fdat.feobj[1].transforms) do
    print('feobj[1]', k, v)
  end
  for k, v in pairs(fdat.feobj[2].transforms) do
    print('feobj[2]', k, v)
  end

  tester:asserteq(fdat.feobj[1].transforms.input, 'input')
  tester:asserteq(fdat.feobj[1].transforms.output, 'float virtualOutput1')

  tester:asserteq(fdat.feobj[2].transforms.input1, 'input1')
  tester:asserteq(fdat.feobj[2].transforms.input2, 'virtualOutput1')
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

