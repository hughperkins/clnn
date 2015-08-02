require 'nngraph'
require 'clnn'

local ngh = require('nodeGraphHelper')
local gh = require('graphHelper')
local fusion = require('fusion')

local fusiontests = {}

function nngraph.Node:graphNodeName()
  if self.data.id ~= nil then
    res = tostring(self.data.id)
    local dat = self.data
    if dat.module ~= nil then
      local mod = dat.module
      res = res .. ' ' .. torch.type(mod)
      if mod.numInputs ~= nil then
        if dat.numVirtualOutputs ~= nil and dat.numVirtualOutputs > 0 then
          res = res .. ' ' .. mod.numInputs.. ' -> (' .. dat.numVirtualOutputs .. ')' .. ' -> ' .. mod.numOutputs
        else
          res = res .. ' ' .. mod.numInputs .. ' -> ' .. mod.numOutputs
        end
      end
    end
    return res
  end
  if self.data.annotations.name then
    return self.data.annotations.name .. ' (' .. self.id .. ')'
  else
    return 'Node' .. self.id
  end
end

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
  local out = nn.Identity()({n2})

  ngh.walkAddParents(out)
  x = ngh.invertGraph(out)
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

  tester:asserteq(#x.children[1].data.outputs, 1)
  tester:asserteq(x.children[1].data.outputs[1].child, x.children[1].children[1])
  tester:asserteq(x.children[1].data.outputs[1].outputIdx, 1)
  tester:asserteq(x.children[1].data.outputs[1].InputIdx, 1)

  fusion.generateKernels(x)
end

function fusiontests.testOutputsTwoOutput()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.Exp()(n1)
  local n3 = nn.Tanh()(n1)
  local out = nn.Identity()({n2, n3})

  ngh.nodeSetName(x, 'x')
  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')
  ngh.nodeSetName(n3, 'n3')
  ngh.nodeSetName(out, 'out')

  ngh.walkAddParents(out)
  ngh.walkRemoveBidirectional(out)
  tester:asserteq(ngh.walkValidate(out), true)
  x = ngh.invertGraph(out)
  ngh.walkAddDataIds(x)
  tester:asserteq(ngh.walkValidate(x), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.walkValidate(x), true)
  if os.getenv('TESTS') ~= nil then ngh.dot(x, '', 'x') end

  tester:asserteq(#n1.data.outputs, 2) 
  tester:asserteq(#n1.data.outputs[1].outputIdx, 1) 
  tester:asserteq(#n1.data.outputs[1].child, n2)
  tester:asserteq(#n1.data.outputs[2].outputIdx, 1) 
  tester:asserteq(#n1.data.outputs[2].child, n3)
end

function fusiontests.testFuseTanhSigmoid()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.Tanh()(n1)
  local out = nn.Identity()({n2})

  ngh.nodeSetName(x, 'x')
  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')
  ngh.nodeSetName(out, 'out')

  ngh.walkAddParents(out)
  ngh.walkRemoveBidirectional(out)
  tester:asserteq(ngh.walkValidate(out), true)
  x = ngh.invertGraph(out)
  ngh.walkAddDataIds(x)
  tester:asserteq(ngh.walkValidate(x), true)

  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.count(x), 4)
  if os.getenv('TESTS') ~= nil then ngh.dot(x, '', 'xbefore') end
  tester:asserteq(ngh.walkValidate(x), true)
  fusion.doFuse(x)
  if os.getenv('TESTS') ~= nil then ngh.dot(x, '', 'xafter') end
  tester:asserteq(ngh.walkValidate(x), true)
  tester:asserteq(ngh.count(x), 3)

  tester:asserteq(torch.type(x.data.module), 'nn.Identity')

  tester:asserteq(ngh.nodeGetName(x), 'x')
  tester:asserteq(#x.children, 1)
  tester:asserteq(#x.children[1].children, 1)
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

  tester:asserteq(#fused.data.outputs, 1)
  print('x.child.out[1]', fused.data.outputs[1].child.data.module)
  print('out', out.data.module)
  tester:asserteq(fused.data.outputs[1].child, out)
  tester:asserteq(fused.data.outputs[1].outputIdx, 1)

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
    ngh.walkApply(x, function(node)
      local dat = node.data
      if dat.feobj ~= nil then
        for i, feobj in ipairs(dat.feobj) do
          for k, v in pairs(feobj.transforms) do
            print('feobj[' .. i .. ']', k, v.src .. v.idx)
          end
          print('')
        end
      end
    end)
  end
  ngh.dot(x, '', 'testInputOrderThreeWayAfter')

  tester:asserteq(ngh.getLinkPos(x1.children[1].parents, x1), 1)
  tester:asserteq(ngh.getLinkPos(x2.children[1].parents, x2), 2)
  tester:asserteq(ngh.getLinkPos(x3.children[1].parents, x3), 3)

  fusion.generateKernels(x)
end

function fusiontests.testClonedOutput()
  local name = 'testClonedOutput'

  local x = nn.Identity()()
  local n1 = nn.Tanh()(x)
  local n2 = nn.Abs()(n1)
  local n3 = nn.Sigmoid()(n1)
  local n4 = nn.CAddTable()({n2, n3})

  ngh.nodeSetName(x, 'x')

  ngh.nodeSetName(n1, 'n1')
  ngh.nodeSetName(n2, 'n2')
  ngh.nodeSetName(n3, 'n3')
  ngh.nodeSetName(n4, 'n4')

  ngh.walkAddParents(n4)
  x = ngh.invertGraph(n4)
  ngh.walkRemoveBidirectional(x)
  ngh.walkAddDataIds(x)

  if os.getenv('TESTS') ~= nil then ngh.dot(x, '', name .. 'Orig') end
  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.count(x), 5)
  tester:asserteq(ngh.walkValidate(x), true)
  if os.getenv('TESTS') ~= nil then ngh.dot(x, '', name .. 'Before') end
  tester:asserteq(ngh.walkValidate(x), true)
  local xold = ngh.walkClone(x)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraph(x)
  local it = 0
  if os.getenv('TESTS') ~= nil then ngh.dot(x, '', 'xit' .. it) end
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(ngh.walkValidate(x), true)
    if os.getenv('TESTS') ~= nil then ngh.dot(x, '', 'xit' .. it) end
--    fusion.generateKernels(x)
    ngh.printGraphWithLinks(x)
    ngh.walkApply(x, function(node)
      local dat = node.data
      if dat.feobj ~= nil then
        for i, feobj in ipairs(dat.feobj) do
          for k, v in pairs(feobj.transforms) do
            print('feobj[' .. i .. ']', k, v.src .. v.idx)
          end
          print('')
        end
      end
    end)
  end
  if os.getenv('TESTS') ~= nil then ngh.dot(x, '', name .. 'After') end

  tester:asserteq(x.children[1].data.module.numInputs, 2)

--  tester:asserteq(ngh.getLinkPos(x1.children[1].parents, x1), 1)
--  tester:asserteq(ngh.getLinkPos(x2.children[1].parents, x2), 2)
--  tester:asserteq(ngh.getLinkPos(x3.children[1].parents, x3), 3)

--  fusion.generateKernels(x)
end

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
--    fusion.generateKernels(x)
    ngh.printGraphWithLinks(x)
--    if it >= 8 then
--      os.exit(0)
--    end
--    fusion.generateKernels(x)
--    ngh.walkApply(x, function(node)
--      local dat = node.data
--      if dat.feobj ~= nil then
--        for i, feobj in ipairs(dat.feobj) do
--          for k, v in pairs(feobj.transforms) do
--            print('feobj[' .. i .. ']', k, v.src .. v.idx)
--          end
--          print('')
--        end
--      end
--    end)
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

--  tester:asserteq(fdat.feobj[1].transforms.input1.src, 'input')
--  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')

--  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'virtualOutput')
--  tester:asserteq(fdat.feobj[2].transforms.input2.src, 'input')
--  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'output')

  fusion.generateKernels(x)
end

function fusiontests.forward1()
  local x = nn.Identity()()
  local n1 = nn.Tanh()(x)

  local g = nn.gModule({x}, {n1}):cl()

  local input = torch.ClTensor(5,3):uniform()
  local outputbefore = g:forward(input)
  print('outputbefore', outputbefore)

  local x = ngh.nnGraphToNgh(g)
  
  tester:asserteq(ngh.walkValidate(x), true)
  fusion.walkConvertToApply(x)
  fusion.generateKernels(x)
  tester:asserteq(ngh.walkValidate(x), true)

  local g2 = ngh.nghToNnGraph(x)
  local outputafter = g2:forward(input)
  print('outputafter', outputafter)

  diff = (outputafter - outputbefore):abs():sum()
  assert(diff == 0)
end

function fusiontests.forward2()
  local x = nn.Identity()()
  local n1 = nn.Tanh()(x)
  local n2 = nn.Sigmoid()(n1)

  local g = nn.gModule({x}, {n2}):cl()

  local input = torch.ClTensor(5,3):uniform()
  local outputbefore = g:forward(input)
  print('outputbefore', outputbefore)

  local x = ngh.nnGraphToNgh(g)
  
  tester:asserteq(ngh.walkValidate(x), true)
  fusion.walkConvertToApply(x)
  while fusion.doFuseIteration(x) do
  end
  fusion.generateKernels(x)
  tester:asserteq(ngh.walkValidate(x), true)
  ngh.printGraphWithLinks(x)

  local g2 = ngh.nghToNnGraph(x)
  if os.getenv('TESTS') ~= nil then
    graph.dot(g2.fg, '', 'g2')
  end
  local outputafter = g2:forward(input)
  print('outputafter', outputafter)

  diff = (outputafter - outputbefore):abs():sum()
  assert(diff == 0)
end

function fusiontests.forward2inputs()
  local x = nn.Identity()()
  local x1, x2 = x:split(2)
  local n1 = nn.CAddTable()({x1, x2})

  local g = nn.gModule({x}, {n1}):cl()

  local input1 = torch.ClTensor(5,3):uniform()
  local input2 = torch.ClTensor(5,3):uniform()
  local inputs = {input1, input2}
  local outputbefore = g:forward(inputs)
  print('outputbefore', outputbefore)

  local x = ngh.nnGraphToNgh(g)
  
  tester:asserteq(ngh.walkValidate(x), true)
  fusion.walkConvertToApply(x)
  while fusion.doFuseIteration(x) do
  end
  ngh.printGraphWithLinks(x)
  fusion.generateKernels(x)
  tester:asserteq(ngh.walkValidate(x), true)

  local g2 = ngh.nghToNnGraph(x)
  if os.getenv('TESTS') ~= nil then
    graph.dot(g2.fg, '', 'g2')
  end
  local outputafter = g2:forward(inputs)
  print('outputafter', outputafter)

  diff = (outputafter - outputbefore):abs():sum()
  assert(diff == 0)
end

function fusiontests.forward2inputs2()
  local x = nn.Identity()()
  local x1, x2 = x:split(2)
  local n1 = nn:Tanh()(x1)
  local n2 = nn:Sigmoid()(x2)
  local n3 = nn.CAddTable()({n1, n2})

  local g = nn.gModule({x}, {n3}):cl()

  local input1 = torch.ClTensor(5,3):uniform()
  local input2 = torch.ClTensor(5,3):uniform()
  local inputs = {input1, input2}
  local outputbefore = g:forward(inputs)
  print('outputbefore', outputbefore)

  local x = ngh.nnGraphToNgh(g)
  
  tester:asserteq(ngh.walkValidate(x), true)
  fusion.walkConvertToApply(x)
  while fusion.doFuseIteration(x) do
  end
  fusion.generateKernels(x)
  ngh.printGraphWithLinks(x)
  tester:asserteq(ngh.walkValidate(x), true)

  local g2 = ngh.nghToNnGraph(x)
  if os.getenv('TESTS') ~= nil then
    graph.dot(g2.fg, '', 'g2')
  end
  local outputafter = g2:forward(inputs)
  print('outputafter', outputafter)

  diff = (outputafter - outputbefore):abs():sum()
  assert(diff == 0)
end

function fusiontests.forwardThreeWay()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.Tanh()({x2})
  local n2 = nn.CAddTable()({x1, n1})
  local n3 = nn.CMulTable()({n2, x3})

  local g = nn.gModule({x}, {n3}):cl()

  local input1 = torch.ClTensor(5,3):uniform()
  local input2 = torch.ClTensor(5,3):uniform()
  local input3 = torch.ClTensor(5,3):uniform()
  local inputs = {input1, input2, input3}
  local outputbefore = g:forward(inputs)
  print('outputbefore', outputbefore)

  local x = ngh.nnGraphToNgh(g)
  
  tester:asserteq(ngh.walkValidate(x), true)
  fusion.walkConvertToApply(x)
  while fusion.doFuseIteration(x) do
  end
  fusion.generateKernels(x)
  ngh.printGraphWithLinks(x)
  tester:asserteq(ngh.walkValidate(x), true)

  local g2 = ngh.nghToNnGraph(x)
  if os.getenv('TESTS') ~= nil then
    graph.dot(g2.fg, '', 'g2')
  end
  local outputafter = g2:forward(inputs)
  print('outputafter', outputafter)

  diff = (outputafter - outputbefore):abs():sum()
  assert(diff == 0)
end

function fusiontests.forwardLSTMNofuse()
  require('test.lstm.OneHot')
  local LSTM = require('test.lstm.LSTM')
  lstm = LSTM.lstm(65, 128, 1, 0):cl()
  if os.getenv('TESTS') ~= nil then
    graph.dot(lstm.fg, '', 'lstm.g')
  end

  local input1 = torch.ClTensor(50):fill(4)
  local input2 = torch.ClTensor(50, 128):fill(2)
  local input3 = torch.ClTensor(50, 128):fill(3)
  local inputs = {input1, input2, input3}
  local outputbefore = lstm:forward(inputs)
  print('output', outputbefore)

  x = ngh.nnGraphToNgh(lstm)
  ngh.printGraph(x)
  if os.getenv('TESTS') ~= nil then
    ngh.dot(x, '', 'lstm1')
  end

  tester:asserteq(ngh.walkValidate(x), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.walkValidate(x), true)
  fusion.generateKernels(x)

  local g2 = ngh.nghToNnGraph(x)
  if os.getenv('TESTS') ~= nil then
    graph.dot(g2.fg, '', 'g2')
  end
  local outputafter = g2:forward(inputs)
  print('outputafter', outputafter)

  for o=1, #outputbefore do
    diff = (outputafter[o] - outputbefore[o]):abs():sum()
    assert(diff == 0)
  end

--  local it = 0
--  print('it ' .. it .. ' ======================')
--  ngh.dot(x, '', 'xit' .. it)
--  while fusion.doFuseIteration(x) do
--    it = it + 1
--    print('it ' .. it .. ' ======================')
--    tester:asserteq(ngh.walkValidate(x), true)
--    ngh.dot(x, '', 'xit' .. it)
--  end

--  fusion.generateKernels(x)
end

function fusiontests.forwardLSTMFused()
  require('test.lstm.OneHot')
  local LSTM = require('test.lstm.LSTM')
  lstm = LSTM.lstm(65, 128, 1, 0):cl()
  if os.getenv('TESTS') ~= nil then
    graph.dot(lstm.fg, '', 'lstm.g')
  end

  local input1 = torch.ClTensor(50):fill(4)
  local input2 = torch.ClTensor(50, 128):fill(2)
  local input3 = torch.ClTensor(50, 128):fill(3)
  local inputs = {input1, input2, input3}
  local outputbefore = lstm:forward(inputs)
  print('output', outputbefore)

  x = ngh.nnGraphToNgh(lstm)
  ngh.walkApply(x, function(node)
    ngh.nodeSetName(node, 'node ' .. node.data.id)
  end)
  ngh.printGraph(x)
  if os.getenv('TESTS') ~= nil then ngh.dot(x, '', 'lstm1') end

  tester:asserteq(ngh.walkValidate(x), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.walkValidate(x), true)

  it = 0
  print('it ' .. it .. ' ======================')
  if os.getenv('TESTS') ~= nil then ngh.dot(x, '', 'xit' .. it) end
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(ngh.walkValidate(x), true)
    if os.getenv('TESTS') ~= nil then ngh.dot(x, '', 'xit' .. it) end
  end

  fusion.generateKernels(x)

  ngh.printGraphWithLinks(x)
  local g2 = ngh.nghToNnGraph(x)
  if os.getenv('TESTS') ~= nil then graph.dot(g2.fg, '', 'g2') end
  local outputafter = g2:forward(inputs)
  print('outputafter', outputafter)

  for o=1, #outputbefore do
    diff = (outputafter[o] - outputbefore[o]):abs():sum()
    assert(diff == 0)
  end

--  local it = 0
--  print('it ' .. it .. ' ======================')
--  ngh.dot(x, '', 'xit' .. it)
--  while fusion.doFuseIteration(x) do
--    it = it + 1
--    print('it ' .. it .. ' ======================')
--    tester:asserteq(ngh.walkValidate(x), true)
--    ngh.dot(x, '', 'xit' .. it)
--  end

--  fusion.generateKernels(x)
end

function fusiontests.testLSTM()
  require('test.lstm.OneHot')
  local LSTM = require('test.lstm.LSTM')
  lstm = LSTM.lstm(65, 128, 2, 0)
  graph.dot(lstm.fg, '', 'lstm.g')
  x = ngh.nnGraphToNgh(lstm)
  ngh.printGraph(x)
  ngh.dot(x, '', 'lstm1')

  tester:asserteq(ngh.walkValidate(x), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(ngh.walkValidate(x), true)

  local it = 0
  print('it ' .. it .. ' ======================')
  ngh.dot(x, '', 'xit' .. it)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(ngh.walkValidate(x), true)
    ngh.dot(x, '', 'xit' .. it)
  end

  fusion.generateKernels(x)
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

