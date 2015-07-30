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
  tester:asserteq(n1.data.numVirtualOutputs, 0)
  tester:asserteq(#n1.data.feobj, 1)
  tester:asserteq(#n1.data.beobj, 1)
  tester:asserteq(n1.data.feobj[1].template, '{{output1}} = tanh({{input1}});')
--  tester:asserteq(n1.data.beobj[1].template, '{{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});')
  tester:asserteq(n1.data.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(n1.data.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(n1.data.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(n1.data.feobj[1].transforms.output1.idx, 1)
  fusion.generateKernels(x)
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
  tester:asserteq(n1.data.numVirtualOutputs, 0)
  tester:asserteq(#n1.data.feobj, 1)
  tester:asserteq(#n1.data.beobj, 1)
  tester:asserteq(n1.data.feobj[1].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
--  tester:asserteq(n1.data.beobj[1].template, '{{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});')
  tester:asserteq(n1.data.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(n1.data.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(n1.data.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(n1.data.feobj[1].transforms.output1.idx, 1)

  fusion.generateKernels(x)
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
  tester:asserteq(n2.data.numVirtualOutputs, 0)
  tester:asserteq(#n2.data.feobj, 1)
  tester:asserteq(#n2.data.beobj, 1)
  tester:asserteq(n2.data.feobj[1].template, '{{output1}} = tanh({{input1}});')
--  tester:asserteq(n2.data.beobj[1].template, '{{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});')
  tester:asserteq(n2.data.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(n2.data.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(n2.data.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(n2.data.feobj[1].transforms.output1.idx, 1)

  tester:asserteq(torch.type(n1.data.module), 'nn.Apply')
  tester:asserteq(n1.data.numVirtualOutputs, 0)
  tester:asserteq(#n1.data.feobj, 1)
  tester:asserteq(#n1.data.beobj, 1)
  tester:asserteq(n1.data.feobj[1].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
--  tester:asserteq(n1.data.beobj[1].template, '{{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});')
  tester:asserteq(n1.data.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(n1.data.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(n1.data.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(n1.data.feobj[1].transforms.output1.idx, 1)

  fusion.generateKernels(x)
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
  tester:asserteq(fdat.feobj[1].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
  tester:asserteq(fdat.feobj[2].template, '{{output1}} = tanh({{input1}});')
  tester:asserteq(fdat.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[1].transforms.output1.idx, 1)
  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[2].transforms.output1.idx, 1)

  fusion.generateKernels(x)
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
  tester:asserteq(fdat.feobj[1].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
  tester:asserteq(fdat.feobj[2].template, '{{output1}} = tanh({{input1}});')
  tester:asserteq(fdat.feobj[3].template, '{{output1}} = exp({{input1}});')

  tester:asserteq(fdat.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[1].transforms.output1.idx, 1)

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.output1.idx, 2)

  tester:asserteq(fdat.feobj[3].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[3].transforms.input1.idx, 2)
  tester:asserteq(fdat.feobj[3].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[3].transforms.output1.idx, 1)

  fusion.generateKernels(x)
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
  tester:asserteq(ngh.count(x), 3)
  tester:asserteq(ngh.walkValidate(x), true)
  fusion.doFuse(x)
  tester:asserteq(ngh.walkValidate(x), true)
  tester:asserteq(ngh.count(x), 2)

  tester:asserteq(torch.type(x.data.module), 'nn.Identity')

  local fused = x.children[1]
  local fdat = fused.data
  tester:asserteq(ngh.nodeGetName(fused), 'n2.n1')
  tester:asserteq(#fdat.feobj, 2)
  tester:asserteq(fdat.feobj[1].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
  tester:asserteq(fdat.feobj[2].template, '{{output1}} = {{input1}} + {{input2}};')

  for i, feobj in ipairs(fdat.feobj) do
    for k, v in pairs(feobj.transforms) do
      print('feobj[' .. i .. ']', k, v)
    end
  end

  tester:asserteq(fdat.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[1].transforms.output1.idx, 1)

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[2].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[2].transforms.input2.idx, 2)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[2].transforms.output1.idx, 1)

  fusion.generateKernels(x)
end

function fusiontests.testApplyConvertMultiInputAdd()
  local x = nn.Identity()()
  local x1, x2 = x:split(2)
  local n1 = nn.Tanh()(x1)
  local n2 = nn.CAddTable()({n1, x2})

  ngh.nodeSetName(x, 'x')
  ngh.nodeSetName(x1, 'x1')
  ngh.nodeSetName(x2, 'x2')
  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')

  ngh.walkAddParents(n2)
  x = ngh.invertGraph(n2)
  ngh.walkRemoveBidirectional(x)
  ngh.walkAddDataIds(x)

  fusion.walkConvertToApply(x)
--  ngh.dot(x, '', 'add')
  tester:asserteq(ngh.count(x), 6)
  tester:asserteq(ngh.walkValidate(x), true)
--  ngh.dot(x, '', 'xold')
  tester:asserteq(ngh.walkValidate(x), true)
--  local xold = ngh.walkClone(x)
--  tester:asserteq(ngh.walkValidate(x), true)
--  ngh.printGraph(x)
  fusion.doFuse(x)
  tester:asserteq(ngh.walkValidate(x), true)
--  ngh.printGraph(x)
--  ngh.dot(x, '', 'xnew')
  tester:asserteq(ngh.count(x), 5)

  tester:asserteq(torch.type(x.data.module), 'nn.Identity')

  local fused = x.children[1].children[1].children[1]
  local fdat = fused.data
  tester:asserteq(ngh.nodeGetName(fused), 'n2.n1')
  tester:asserteq(#fdat.feobj, 2)
  tester:asserteq(fdat.feobj[1].template, '{{output1}} = tanh({{input1}});')
  tester:asserteq(fdat.feobj[2].template, '{{output1}} = {{input1}} + {{input2}};')

  for k, v in pairs(fdat.feobj[1].transforms) do
    print('feobj[1]', k, v)
  end
  for k, v in pairs(fdat.feobj[2].transforms) do
    print('feobj[2]', k, v)
  end

  tester:asserteq(fdat.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[1].transforms.output1.idx, 1)

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[2].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[2].transforms.input2.idx, 2)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[2].transforms.output1.idx, 1)

  tester:asserteq(ngh.getLinkPos(x1.children[1].parents, x1), 1)
  tester:asserteq(ngh.getLinkPos(x2.children[1].parents, x2), 2)

  fusion.generateKernels(x)
end

function fusiontests.testApplyConvertMultiInputAdd3()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.CMulTable()({x1, x2})
  local n2 = nn.CAddTable()({n1, x3})

  ngh.nodeSetName(x, 'x')
  ngh.nodeSetName(x1, 'x1')
  ngh.nodeSetName(x2, 'x2')
  ngh.nodeSetName(x3, 'x3')
  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')

  ngh.walkAddParents(n2)
  x = ngh.invertGraph(n2)
  ngh.walkRemoveBidirectional(x)
  ngh.walkAddDataIds(x)

  fusion.walkConvertToApply(x)
  ngh.dot(x, '', 'add')
  tester:asserteq(ngh.count(x), 7)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.dot(x, '', 'xold')
  tester:asserteq(ngh.walkValidate(x), true)
  local xold = ngh.walkClone(x)
  tester:asserteq(ngh.walkValidate(x), true)
--  ngh.printGraph(x)
  fusion.doFuse(x)
  tester:asserteq(ngh.walkValidate(x), true)
--  ngh.printGraph(x)
  ngh.dot(x, '', 'xnew')
  tester:asserteq(ngh.count(x), 6)

  tester:asserteq(torch.type(x.data.module), 'nn.Identity')

  local fused = x.children[1].children[1].children[1]
  local fdat = fused.data
  tester:asserteq(ngh.nodeGetName(fused), 'n2.n1')
  tester:asserteq(#fdat.feobj, 2)
  tester:asserteq(fdat.feobj[1].template, '{{output1}} = {{input1}} * {{input2}};')
  tester:asserteq(fdat.feobj[2].template, '{{output1}} = {{input1}} + {{input2}};')

  for i, feobj in ipairs(fdat.feobj) do
    for k, v in pairs(feobj.transforms) do
      print('feobj[' .. i .. ']', k, v)
    end
  end

  tester:asserteq(fdat.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[1].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.input2.idx, 2)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[1].transforms.output1.idx, 1)

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[2].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[2].transforms.input2.idx, 3)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[2].transforms.output1.idx, 1)

  tester:asserteq(ngh.getLinkPos(x1.children[1].parents, x1), 1)
  tester:asserteq(ngh.getLinkPos(x2.children[1].parents, x2), 2)
  tester:asserteq(ngh.getLinkPos(x3.children[1].parents, x3), 3)

  fusion.generateKernels(x)
end

function fusiontests.testAddTanhMul()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.CAddTable()({x1, x2})
  local n2 = nn.Tanh()({n1})
  local n3 = nn.CMulTable()({n2, x3})

  ngh.nodeSetName(x, 'x')
  ngh.nodeSetName(x1, 'x1')
  ngh.nodeSetName(x2, 'x2')
  ngh.nodeSetName(x3, 'x3')

  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')
  ngh.nodeSetName(n3, 'n3')

  ngh.walkAddParents(n3)
  ngh.dot(n3, '', 'testAddTanhMulBeforeInvert')

  x = ngh.invertGraph(n3)
  ngh.walkRemoveBidirectional(x)
  ngh.walkAddDataIds(x)

  ngh.dot(x, '', 'testAddTanhMulBefore')
  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.count(x), 8)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraph(x)

  local it = 0
  print('it ' .. it .. ' ===============')
  ngh.printGraphWithLinks(x)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ===============')
    tester:asserteq(ngh.walkValidate(x), true)
    ngh.printGraphWithLinks(x)
  end

--  ngh.printGraphWithLinks(x)

  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraph(x)
  ngh.dot(x, '', 'testAddTanhMulAfter')
  tester:asserteq(ngh.count(x), 6)

  local fused = x.children[1].children[1].children[1]
  local fdat = fused.data
  tester:asserteq(ngh.nodeGetName(fused), 'n3.n2.n1')
  tester:asserteq(#fdat.feobj, 3)
  tester:asserteq(fdat.feobj[1].template, '{{output1}} = {{input1}} + {{input2}};')
  tester:asserteq(fdat.feobj[2].template, '{{output1}} = tanh({{input1}});')
  tester:asserteq(fdat.feobj[3].template, '{{output1}} = {{input1}} * {{input2}};')

  for i, feobj in ipairs(fdat.feobj) do
    for k, v in pairs(feobj.transforms) do
      print('feobj[' .. i .. ']', k, v)
    end
  end

  tester:asserteq(fdat.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[1].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.input2.idx, 2)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[1].transforms.output1.idx, 1)

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input1.idx, 1)
--  tester:asserteq(fdat.feobj[2].transforms.input2.src, 'input')
--  tester:asserteq(fdat.feobj[2].transforms.input2.idx, 3)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.output1.idx, 2)

  tester:asserteq(fdat.feobj[3].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[3].transforms.input1.idx, 2)
  tester:asserteq(fdat.feobj[3].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[3].transforms.input2.idx, 3)
  tester:asserteq(fdat.feobj[3].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[3].transforms.output1.idx, 1)

  tester:asserteq(ngh.getLinkPos(x1.children[1].parents, x1), 1)
  tester:asserteq(ngh.getLinkPos(x2.children[1].parents, x2), 2)
  tester:asserteq(ngh.getLinkPos(x3.children[1].parents, x3), 3)

  fusion.generateKernels(x)
end

function fusiontests.testSigMulAdd()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.Sigmoid()({x2})
  local n2 = nn.CMulTable()({x1, n1})
  local n3 = nn.CAddTable()({n2, x3})

  ngh.nodeSetName(x, 'x')
  ngh.nodeSetName(x1, 'x1')
  ngh.nodeSetName(x2, 'x2')
  ngh.nodeSetName(x3, 'x3')

  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')
  ngh.nodeSetName(n3, 'n3')

  ngh.walkAddParents(n3)
  ngh.dot(n3, '', 'testSigMulAddBeforeInvert')

  x = ngh.invertGraph(n3)
  ngh.walkRemoveBidirectional(x)
  ngh.walkAddDataIds(x)

  ngh.dot(x, '', 'testSigMulAddBefore')
  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.count(x), 8)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraph(x)

  local it = 0
  print('it ' .. it .. ' ===============')
  ngh.printGraphWithLinks(x)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ===============')
    tester:asserteq(ngh.walkValidate(x), true)
    ngh.printGraphWithLinks(x)
  end

--  ngh.printGraphWithLinks(x)

  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraph(x)
  ngh.dot(x, '', 'testSigMulAddAfter')
  tester:asserteq(ngh.count(x), 6)

  local fused = x.children[1].children[1].children[1]
  local fdat = fused.data
  tester:asserteq(ngh.nodeGetName(fused), 'n3.n2.n1')
  tester:asserteq(#fdat.feobj, 3)
  tester:asserteq(fdat.feobj[1].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
  tester:asserteq(fdat.feobj[2].template, '{{output1}} = {{input1}} * {{input2}};')
  tester:asserteq(fdat.feobj[3].template, '{{output1}} = {{input1}} + {{input2}};')

  for i, feobj in ipairs(fdat.feobj) do
    for k, v in pairs(feobj.transforms) do
      print('feobj[' .. i .. ']', k, v)
    end
  end

  tester:asserteq(fdat.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.input1.idx, 2)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[1].transforms.output1.idx, 2)

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[2].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[2].transforms.input2.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input2.idx, 2)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.output1.idx, 1)

  tester:asserteq(fdat.feobj[3].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[3].transforms.input1.idx, 1)
  tester:asserteq(fdat.feobj[3].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[3].transforms.input2.idx, 3)
  tester:asserteq(fdat.feobj[3].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[3].transforms.output1.idx, 1)

  tester:asserteq(ngh.getLinkPos(x1.children[1].parents, x1), 1)
  tester:asserteq(ngh.getLinkPos(x2.children[1].parents, x2), 2)
  tester:asserteq(ngh.getLinkPos(x3.children[1].parents, x3), 3)

  fusion.generateKernels(x)
end

function fusiontests.testInputOrderThreeWay()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.Tanh()({x2})
  local n2 = nn.CAddTable()({x1, n1})
  local n3 = nn.CMulTable()({n2, x3})

  ngh.nodeSetName(x, 'x')

  ngh.nodeSetName(x1, 'x1')
  ngh.nodeSetName(x2, 'x2')
  ngh.nodeSetName(x3, 'x3')

  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')
  ngh.nodeSetName(n3, 'n3')

  ngh.walkAddParents(n3)
  x = ngh.invertGraph(n3)
  ngh.walkRemoveBidirectional(x)
  ngh.walkAddDataIds(x)

  ngh.dot(x, '', 'testInputOrderThreeWayadd')
  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.count(x), 8)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.dot(x, '', 'testInputOrderThreeWayBefore')
  tester:asserteq(ngh.walkValidate(x), true)
  local xold = ngh.walkClone(x)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraph(x)
  local it = 0
  ngh.dot(x, '', 'xit' .. it)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(ngh.walkValidate(x), true)
    ngh.dot(x, '', 'xit' .. it)
    fusion.generateKernels(x)
    ngh.printGraphWithLinks(x)
  end
  ngh.dot(x, '', 'testInputOrderThreeWayAfter')

  tester:asserteq(ngh.getLinkPos(x1.children[1].parents, x1), 1)
  tester:asserteq(ngh.getLinkPos(x2.children[1].parents, x2), 2)
  tester:asserteq(ngh.getLinkPos(x3.children[1].parents, x3), 3)
end

if false then
function fusiontests.testApplyCharRnn()
  local x = nn.Identity()()
  local xpre, x1, x2, x3, x4 = x:split(5)
  local n1 = nn.Sigmoid()(x1)
  local n2 = nn.Sigmoid()(x2)
  local n3 = nn.Tanh()(x3)
  local n4 = nn.CMulTable()({xpre, n1})
  local n5 = nn.CMulTable()({n2, n3})
  local n6 = nn.CAddTable()({n4, n5})
  local n7 = nn.Tanh()(n6)
  local n8 = nn.Sigmoid()(x4)
  local n9 = nn.CMulTable()({n7, n8})
  local out = nn.Identity()({n6, n9})

  ngh.nodeSetName(x, 'x')
  ngh.nodeSetName(xpre, 'xpre')
  ngh.nodeSetName(x1, 'x1')
  ngh.nodeSetName(x2, 'x2')
  ngh.nodeSetName(x3, 'x3')
  ngh.nodeSetName(x4, 'x4')
  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')
  ngh.nodeSetName(n3, 'n3')
  ngh.nodeSetName(n4, 'n4')
  ngh.nodeSetName(n5, 'n5')
  ngh.nodeSetName(n6, 'n6')
  ngh.nodeSetName(n7, 'n7')
  ngh.nodeSetName(n8, 'n8')
  ngh.nodeSetName(n9, 'n9')
  ngh.nodeSetName(out, 'out')

  ngh.walkAddParents(n9)
  x = ngh.invertGraph(n9)
  ngh.walkRemoveBidirectional(x)
  ngh.walkAddDataIds(x)

  ngh.dot(x, '', 'add')
  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.count(x), 16)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.dot(x, '', 'xold')
  tester:asserteq(ngh.walkValidate(x), true)
  local xold = ngh.walkClone(x)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraph(x)
  local it = 0
  ngh.dot(x, '', 'xit' .. it)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(ngh.walkValidate(x), true)
    ngh.dot(x, '', 'xit' .. it)
    fusion.generateKernels(x)
    ngh.printGraphWithLinks(x)
    if it >= 8 then
      os.exit(0)
    end
  end
--  fusion.doFuse(x)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraph(x)
  ngh.dot(x, '', 'xnew')
  tester:asserteq(ngh.count(x), 8)

  tester:asserteq(torch.type(x.data.module), 'nn.Identity')

  local fused = x.children[1].children[1].children[1]
  local fdat = fused.data
  tester:asserteq(ngh.nodeGetName(fused), 'n9.n7.n6.n4.n1.n5.n2.n3.n8')
  tester:asserteq(#fdat.feobj, 9)
  tester:asserteq(fdat.feobj[1].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
  tester:asserteq(fdat.feobj[2].template, '{{output1}} = tanh({{input1}});')
  tester:asserteq(fdat.feobj[3].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
  tester:asserteq(fdat.feobj[4].template, '{{output1}} = {{input1}} * {{input2}};')
  tester:asserteq(fdat.feobj[5].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
  tester:asserteq(fdat.feobj[6].template, '{{output1}} = {{input1}} * {{input2}};')
  tester:asserteq(fdat.feobj[7].template, '{{output1}} = {{input1}} + {{input2}};')
  tester:asserteq(fdat.feobj[8].template, '{{output1}} = tanh({{input1}});')
  tester:asserteq(fdat.feobj[9].template, '{{output1}} = {{input1}} * {{input2}};')

  for k, v in pairs(fdat.feobj[1].transforms) do
    print('feobj[1]', k, v)
  end
  for k, v in pairs(fdat.feobj[2].transforms) do
    print('feobj[2]', k, v)
  end

  tester:asserteq(fdat.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'output')

  fusion.generateKernels(x)
end
end

function go()
  nloop = n_loop or nloop
  local oldtype = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  -- initSeed(seed)
  tester = torch.Tester()
  local targettests = fusiontests
  if os.getenv('LIST') ~= nil then
    print('fusiontests', fusiontests)
    os.exit(0)
  end
  if os.getenv('TESTS') ~= nil then
    targettests = {}
    local filter = os.getenv('TESTS')
    for k, v in pairs(fusiontests) do
      if k:find(filter) ~= nil then
        targettests[k] = v
      end
    end
  end
  print('targettests', targettests)
  tester:add(targettests)
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

