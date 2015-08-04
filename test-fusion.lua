require 'nngraph'
require 'clnn'

--local fusibles = require('Fusible')
local gh = require('graphHelper')
local fusion = require('fusion')

local fusiontests = {}

function nngraph.Node:graphNodeName()
  if self.id ~= nil then
    res = tostring(self.id)
    local dat = self
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
  if self.annotations.name then
    return self.annotations.name .. ' (' .. self.id .. ')'
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

  x = nn.Fusible.fromNodes(n1)

--  fusibles.walkAddParents(n1)
--  x = fusibles.invertGraph(n1)
--  fusibles.walkRemoveBidirectional(x)

  fusion.walkConvertToApply(x)
  tester:asserteq(torch.type(x.outputs[1].child.module), 'nn.Apply')
  tester:asserteq(torch.type(x.module), 'nn.Identity')
  tester:asserteq(x.outputs[1].child.numVirtualOutputs, 0)
  n1 = x.outputs[1].child
  tester:asserteq(#n1.feobj, 1)
  tester:asserteq(#n1.beobj, 1)
  tester:asserteq(n1.feobj[1].template, '{{output1}} = tanh({{input1}});')
--  tester:asserteq(n1.beobj[1].template, '{{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});')
  tester:asserteq(n1.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(n1.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(n1.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(n1.feobj[1].transforms.output1.idx, 1)

  fusion.generateKernels(x)
end

function fusiontests.testApplyConvertSigmoid()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)

  fusibles.walkAddParents(n1)
  x = fusibles.invertGraph(n1)
  fusibles.walkRemoveBidirectional(x)
  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(x:walkValidate(), true)

  tester:asserteq(torch.type(n1.module), 'nn.Apply')
  tester:asserteq(torch.type(x.module), 'nn.Identity')
  tester:asserteq(n1.numVirtualOutputs, 0)
  tester:asserteq(#n1.feobj, 1)
  tester:asserteq(#n1.beobj, 1)
  tester:asserteq(n1.feobj[1].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
--  tester:asserteq(n1.beobj[1].template, '{{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});')
  tester:asserteq(n1.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(n1.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(n1.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(n1.feobj[1].transforms.output1.idx, 1)

  fusion.generateKernels(x)
end

function fusiontests.testApplyConvertTanhSigmoid()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.Tanh()(n1)
  local out = nn.Identity()({n2})

  x = nn.Fusible.fromNodes(n2)

  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(x:walkValidate(), true)

  tester:asserteq(torch.type(x.module), 'nn.Identity')

  n1 = x.outputs[1].child
  n2 = n1.outputs[1].child
  tester:asserteq(torch.type(n2.module), 'nn.Apply')
  tester:asserteq(n2.numVirtualOutputs, 0)
  tester:asserteq(#n2.feobj, 1)
  tester:asserteq(#n2.beobj, 1)
  tester:asserteq(n2.feobj[1].template, '{{output1}} = tanh({{input1}});')
--  tester:asserteq(n2.beobj[1].template, '{{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});')
  tester:asserteq(n2.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(n2.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(n2.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(n2.feobj[1].transforms.output1.idx, 1)

  tester:asserteq(torch.type(n1.module), 'nn.Apply')
  tester:asserteq(n1.numVirtualOutputs, 0)
  tester:asserteq(#n1.feobj, 1)
  tester:asserteq(#n1.beobj, 1)
  tester:asserteq(n1.feobj[1].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
--  tester:asserteq(n1.beobj[1].template, '{{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});')
  tester:asserteq(n1.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(n1.feobj[1].transforms.input1.idx, 1)
  tester:asserteq(n1.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(n1.feobj[1].transforms.output1.idx, 1)

  tester:asserteq(#n1.outputs, 1)
  tester:asserteq(n1.outputs[1].child, n2)
  tester:asserteq(n1.outputs[1].outputIdx, 1)
  tester:asserteq(n1.outputs[1].inputIdx, 1)

  fusion.generateKernels(x)
end

function fusiontests.testOutputsTwoOutput()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.Exp()(n1)
  local n3 = nn.Tanh()(n1)
  local out = nn.Identity()({n2, n3})

  fusibles.nodeSetName(x, 'x')
  fusibles.nodeSetName(n1, 'n1')
  fusibles.nodeSetName(n2, 'n2')
  fusibles.nodeSetName(n3, 'n3')
  fusibles.nodeSetName(out, 'out')

  fusibles.walkAddParents(out)
  fusibles.walkRemoveBidirectional(out)
  tester:asserteq(fusibles.walkValidate(out), true)
  x = fusibles.invertGraph(out)
  fusibles.walkAddDataIds(x)
  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(x:walkValidate(), true)
  if os.getenv('TESTS') ~= nil then x:dot('', 'x') end

  tester:asserteq(#n1.outputs, 2) 
  tester:asserteq(#n1.outputs[1].outputIdx, 1) 
  tester:asserteq(#n1.outputs[1].child, n2)
  tester:asserteq(#n1.outputs[2].outputIdx, 1) 
  tester:asserteq(#n1.outputs[2].child, n3)
end

function fusiontests.testFuseTanhSigmoid()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.Tanh()(n1)
  local out = nn.Identity()({n2})

  fusibles.nodeSetName(x, 'x')
  fusibles.nodeSetName(n1, 'n1')
  fusibles.nodeSetName(n2, 'n2')
  fusibles.nodeSetName(out, 'out')

  x = nn.Fusible.fromNodes(out)

  tester:asserteq(x:walkValidate(), true)

  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 4)
  if os.getenv('TESTS') ~= nil then x:dot('', 'xbefore') end
  tester:asserteq(x:walkValidate(), true)
  fusion.doFuse(x)
  if os.getenv('TESTS') ~= nil then x:dot('', 'xafter') end
  tester:asserteq(x:walkValidate(), true)
  tester:asserteq(x:count(), 3)

  tester:asserteq(torch.type(x.module), 'nn.Identity')

  tester:asserteq(x.name, 'x')
  tester:asserteq(#x.outputs, 1)
  tester:asserteq(#x.outputs[1].child.outputs, 1)
  local fused = x.outputs[1].child
  local fdat = fused
  tester:asserteq(fused.name, 'n2.n1')
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

  tester:asserteq(#fused.outputs, 1)
  print('x.child.out[1]', fused.outputs[1].child.module)
  print('out', out.module)
  tester:asserteq(fused.outputs[1].child, out)
  tester:asserteq(fused.outputs[1].outputIdx, 1)

  fusion.generateKernels(x)
end

function fusiontests.testFuseExpTanhSigmoid()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.Tanh()(n1)
  local n3 = nn.Exp()(n2)

  fusibles.nodeSetName(x, 'x')
  fusibles.nodeSetName(n1, 'n1')
  fusibles.nodeSetName(n2, 'n2')
  fusibles.nodeSetName(n3, 'n3')

  fusibles.walkAddParents(n3)
  fusibles.walkRemoveBidirectional(n3)
  x = fusibles.invertGraph(n2)
  fusibles.walkAddDataIds(x)

  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 4)
  fusion.doFuse(x)
  tester:asserteq(x:walkValidate(), true)
  tester:asserteq(x:count(), 2)

  tester:asserteq(torch.type(x.module), 'nn.Identity')

  tester:asserteq(fusibles.nodeGetName(x), 'x')
  tester:asserteq(#x.children, 1)
  tester:asserteq(#x.children[1].children, 0)

  local fused = x.children[1]
  local fdat = fused
  tester:asserteq(fused.name, 'n3.n2.n1')
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

  fusibles.nodeSetName(x, 'x')
  fusibles.nodeSetName(n1, 'n1')
  fusibles.nodeSetName(n2, 'n2')

  fusibles.walkAddParents(n2)
  fusibles.walkRemoveBidirectional(n2)
  x = fusibles.invertGraph(n2)
  fusibles.walkAddDataIds(x)

  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 3)
  tester:asserteq(x:walkValidate(), true)
  fusion.doFuse(x)
  tester:asserteq(x:walkValidate(), true)
  tester:asserteq(x:count(), 2)

  tester:asserteq(torch.type(x.module), 'nn.Identity')

  local fused = x.children[1]
  local fdat = fused
  tester:asserteq(fused.name, 'n2.n1')
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

  fusibles.nodeSetName(x, 'x')
  fusibles.nodeSetName(x1, 'x1')
  fusibles.nodeSetName(x2, 'x2')
  fusibles.nodeSetName(n1, 'n1')
  fusibles.nodeSetName(n2, 'n2')

  fusibles.walkAddParents(n2)
  x = fusibles.invertGraph(n2)
  fusibles.walkRemoveBidirectional(x)
  fusibles.walkAddDataIds(x)

  fusion.walkConvertToApply(x)
--  x:dot('', 'add')
  tester:asserteq(x:count(), 6)
  tester:asserteq(x:walkValidate(), true)
--  x:dot('', 'xold')
  tester:asserteq(x:walkValidate(), true)
--  local xold = fusibles.walkClone(x)
--  tester:asserteq(x:walkValidate(), true)
--  fusibles.printGraph(x)
  fusion.doFuse(x)
  tester:asserteq(x:walkValidate(), true)
--  fusibles.printGraph(x)
--  x:dot('', 'xnew')
  tester:asserteq(x:count(), 5)

  tester:asserteq(torch.type(x.module), 'nn.Identity')

  local fused = x.children[1].children[1].children[1]
  local fdat = fused
  tester:asserteq(fused.name, 'n2.n1')
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

  tester:asserteq(fusibles.getLinkPos(x1.children[1].parents, x1), 1)
  tester:asserteq(fusibles.getLinkPos(x2.children[1].parents, x2), 2)

  fusion.generateKernels(x)
end

function fusiontests.testApplyConvertMultiInputAdd3()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.CMulTable()({x1, x2})
  local n2 = nn.CAddTable()({n1, x3})

  fusibles.nodeSetName(x, 'x')
  fusibles.nodeSetName(x1, 'x1')
  fusibles.nodeSetName(x2, 'x2')
  fusibles.nodeSetName(x3, 'x3')
  fusibles.nodeSetName(n1, 'n1')
  fusibles.nodeSetName(n2, 'n2')

  fusibles.walkAddParents(n2)
  x = fusibles.invertGraph(n2)
  fusibles.walkRemoveBidirectional(x)
  fusibles.walkAddDataIds(x)

  fusion.walkConvertToApply(x)
  x:dot('', 'add')
  tester:asserteq(x:count(), 7)
  tester:asserteq(x:walkValidate(), true)
  x:dot('', 'xold')
  tester:asserteq(x:walkValidate(), true)
  local xold = fusibles.walkClone(x)
  tester:asserteq(x:walkValidate(), true)
--  fusibles.printGraph(x)
  fusion.doFuse(x)
  tester:asserteq(x:walkValidate(), true)
--  fusibles.printGraph(x)
  x:dot('', 'xnew')
  tester:asserteq(x:count(), 6)

  tester:asserteq(torch.type(x.module), 'nn.Identity')

  local fused = x.children[1].children[1].children[1]
  local fdat = fused
  tester:asserteq(fused.name, 'n2.n1')
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

  tester:asserteq(fusibles.getLinkPos(x1.children[1].parents, x1), 1)
  tester:asserteq(fusibles.getLinkPos(x2.children[1].parents, x2), 2)
  tester:asserteq(fusibles.getLinkPos(x3.children[1].parents, x3), 3)

  fusion.generateKernels(x)
end

function fusiontests.testAddTanhMul()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.CAddTable()({x1, x2})
  local n2 = nn.Tanh()({n1})
  local n3 = nn.CMulTable()({n2, x3})

  fusibles.nodeSetName(x, 'x')
  fusibles.nodeSetName(x1, 'x1')
  fusibles.nodeSetName(x2, 'x2')
  fusibles.nodeSetName(x3, 'x3')

  fusibles.nodeSetName(n1, 'n1')
  fusibles.nodeSetName(n2, 'n2')
  fusibles.nodeSetName(n3, 'n3')

  fusibles.walkAddParents(n3)
  fusibles.dot(n3, '', 'testAddTanhMulBeforeInvert')

  x = fusibles.invertGraph(n3)
  fusibles.walkRemoveBidirectional(x)
  fusibles.walkAddDataIds(x)

  x:dot('', 'testAddTanhMulBefore')
  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 8)
  tester:asserteq(x:walkValidate(), true)
  fusibles.printGraph(x)

  local it = 0
  print('it ' .. it .. ' ===============')
  fusibles.printGraphWithLinks(x)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ===============')
    tester:asserteq(x:walkValidate(), true)
    fusibles.printGraphWithLinks(x)
  end

--  fusibles.printGraphWithLinks(x)

  tester:asserteq(x:walkValidate(), true)
  fusibles.printGraph(x)
  x:dot('', 'testAddTanhMulAfter')
  tester:asserteq(x:count(), 6)

  local fused = x.children[1].children[1].children[1]
  local fdat = fused
  tester:asserteq(fused.name, 'n3.n2.n1')
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

  tester:asserteq(fusibles.getLinkPos(x1.children[1].parents, x1), 1)
  tester:asserteq(fusibles.getLinkPos(x2.children[1].parents, x2), 2)
  tester:asserteq(fusibles.getLinkPos(x3.children[1].parents, x3), 3)

  fusion.generateKernels(x)
end

function fusiontests.testSigMulAdd()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.Sigmoid()({x2})
  local n2 = nn.CMulTable()({x1, n1})
  local n3 = nn.CAddTable()({n2, x3})

  fusibles.nodeSetName(x, 'x')
  fusibles.nodeSetName(x1, 'x1')
  fusibles.nodeSetName(x2, 'x2')
  fusibles.nodeSetName(x3, 'x3')

  fusibles.nodeSetName(n1, 'n1')
  fusibles.nodeSetName(n2, 'n2')
  fusibles.nodeSetName(n3, 'n3')

  fusibles.walkAddParents(n3)
  fusibles.dot(n3, '', 'testSigMulAddBeforeInvert')

  x = fusibles.invertGraph(n3)
  fusibles.walkRemoveBidirectional(x)
  fusibles.walkAddDataIds(x)

  x:dot('', 'testSigMulAddBefore')
  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 8)
  tester:asserteq(x:walkValidate(), true)
  fusibles.printGraph(x)

  local it = 0
  print('it ' .. it .. ' ===============')
  fusibles.printGraphWithLinks(x)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ===============')
    tester:asserteq(x:walkValidate(), true)
    fusibles.printGraphWithLinks(x)
  end

--  fusibles.printGraphWithLinks(x)

  tester:asserteq(x:walkValidate(), true)
  fusibles.printGraph(x)
  x:dot('', 'testSigMulAddAfter')
  tester:asserteq(x:count(), 6)

  local fused = x.children[1].children[1].children[1]
  local fdat = fused
  tester:asserteq(fused.name, 'n3.n2.n1')
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

  tester:asserteq(fusibles.getLinkPos(x1.children[1].parents, x1), 1)
  tester:asserteq(fusibles.getLinkPos(x2.children[1].parents, x2), 2)
  tester:asserteq(fusibles.getLinkPos(x3.children[1].parents, x3), 3)

  fusion.generateKernels(x)
end

function fusiontests.testInputOrderThreeWay()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.Tanh()({x2})
  local n2 = nn.CAddTable()({x1, n1})
  local n3 = nn.CMulTable()({n2, x3})

  fusibles.nodeSetName(x, 'x')

  fusibles.nodeSetName(x1, 'x1')
  fusibles.nodeSetName(x2, 'x2')
  fusibles.nodeSetName(x3, 'x3')

  fusibles.nodeSetName(n1, 'n1')
  fusibles.nodeSetName(n2, 'n2')
  fusibles.nodeSetName(n3, 'n3')

  fusibles.walkAddParents(n3)
  x = fusibles.invertGraph(n3)
  fusibles.walkRemoveBidirectional(x)
  fusibles.walkAddDataIds(x)

  x:dot('', 'testInputOrderThreeWayadd')
  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 8)
  tester:asserteq(x:walkValidate(), true)
  x:dot('', 'testInputOrderThreeWayBefore')
  tester:asserteq(x:walkValidate(), true)
  local xold = fusibles.walkClone(x)
  tester:asserteq(x:walkValidate(), true)
  fusibles.printGraph(x)
  local it = 0
  x:dot('', 'xit' .. it)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(x:walkValidate(), true)
    x:dot('', 'xit' .. it)
    fusion.generateKernels(x)
    fusibles.printGraphWithLinks(x)
    fusibles.walkApply(x, function(node)
      local dat = node
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
  x:dot('', 'testInputOrderThreeWayAfter')

  tester:asserteq(fusibles.getLinkPos(x1.children[1].parents, x1), 1)
  tester:asserteq(fusibles.getLinkPos(x2.children[1].parents, x2), 2)
  tester:asserteq(fusibles.getLinkPos(x3.children[1].parents, x3), 3)

  fusion.generateKernels(x)
end

function fusiontests.testClonedOutput()
  local name = 'testClonedOutput'

  local x = nn.Identity()()
  local n1 = nn.Tanh()(x)
  local n2 = nn.Abs()(n1)
  local n3 = nn.Sigmoid()(n1)
  local n4 = nn.CAddTable()({n2, n3})

  fusibles.nodeSetName(x, 'x')

  fusibles.nodeSetName(n1, 'n1')
  fusibles.nodeSetName(n2, 'n2')
  fusibles.nodeSetName(n3, 'n3')
  fusibles.nodeSetName(n4, 'n4')

  fusibles.walkAddParents(n4)
  x = fusibles.invertGraph(n4)
  fusibles.walkRemoveBidirectional(x)
  fusibles.walkAddDataIds(x)

  if os.getenv('TESTS') ~= nil then x:dot('', name .. 'Orig') end
  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 5)
  tester:asserteq(x:walkValidate(), true)
  if os.getenv('TESTS') ~= nil then x:dot('', name .. 'Before') end
  tester:asserteq(x:walkValidate(), true)
  local xold = fusibles.walkClone(x)
  tester:asserteq(x:walkValidate(), true)
  fusibles.printGraph(x)
  local it = 0
  if os.getenv('TESTS') ~= nil then x:dot('', 'xit' .. it) end
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(x:walkValidate(), true)
    if os.getenv('TESTS') ~= nil then x:dot('', 'xit' .. it) end
--    fusion.generateKernels(x)
    fusibles.printGraphWithLinks(x)
    fusibles.walkApply(x, function(node)
      local dat = node
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
  if os.getenv('TESTS') ~= nil then x:dot('', name .. 'After') end

  tester:asserteq(x.children[1].module.numInputs, 2)

--  tester:asserteq(fusibles.getLinkPos(x1.children[1].parents, x1), 1)
--  tester:asserteq(fusibles.getLinkPos(x2.children[1].parents, x2), 2)
--  tester:asserteq(fusibles.getLinkPos(x3.children[1].parents, x3), 3)

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

  fusibles.nodeSetName(x, 'x')
  fusibles.nodeSetName(xpre, 'xpre')
  fusibles.nodeSetName(x1, 'x1')
  fusibles.nodeSetName(x2, 'x2')
  fusibles.nodeSetName(x3, 'x3')
  fusibles.nodeSetName(x4, 'x4')
  fusibles.nodeSetName(n1, 'n1')
  fusibles.nodeSetName(n2, 'n2')
  fusibles.nodeSetName(n3, 'n3')
  fusibles.nodeSetName(n4, 'n4')
  fusibles.nodeSetName(n5, 'n5')
  fusibles.nodeSetName(n6, 'n6')
  fusibles.nodeSetName(n7, 'n7')
  fusibles.nodeSetName(n8, 'n8')
  fusibles.nodeSetName(n9, 'n9')
  fusibles.nodeSetName(out, 'out')

  fusibles.walkAddParents(n9)
  x = fusibles.invertGraph(n9)
  fusibles.walkRemoveBidirectional(x)
  fusibles.walkAddDataIds(x)

  x:dot('', 'add')
  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 16)
  tester:asserteq(x:walkValidate(), true)
  x:dot('', 'xold')
  tester:asserteq(x:walkValidate(), true)
  local xold = fusibles.walkClone(x)
  tester:asserteq(x:walkValidate(), true)
  fusibles.printGraph(x)
  local it = 0
  x:dot('', 'xit' .. it)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(x:walkValidate(), true)
    x:dot('', 'xit' .. it)
--    fusion.generateKernels(x)
    fusibles.printGraphWithLinks(x)
--    if it >= 8 then
--      os.exit(0)
--    end
--    fusion.generateKernels(x)
--    fusibles.walkApply(x, function(node)
--      local dat = node
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
  tester:asserteq(x:walkValidate(), true)
  fusibles.printGraph(x)
  x:dot('', 'xnew')
  tester:asserteq(x:count(), 8)

  tester:asserteq(torch.type(x.module), 'nn.Identity')

  local fused = x.children[1].children[1].children[1]
  local fdat = fused
  tester:asserteq(fused.name, 'n9.n7.n6.n4.n1.n5.n2.n3.n8')
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

  local x = fusibles.nnGraphToNgh(g)
  
  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  fusion.generateKernels(x)
  tester:asserteq(x:walkValidate(), true)

  local g2 = fusibles.nghToNnGraph(x)
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

  local x = fusibles.nnGraphToNgh(g)
  
  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  while fusion.doFuseIteration(x) do
  end
  fusion.generateKernels(x)
  tester:asserteq(x:walkValidate(), true)
  fusibles.printGraphWithLinks(x)

  local g2 = fusibles.nghToNnGraph(x)
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

  local x = fusibles.nnGraphToNgh(g)
  
  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  while fusion.doFuseIteration(x) do
  end
  fusibles.printGraphWithLinks(x)
  fusion.generateKernels(x)
  tester:asserteq(x:walkValidate(), true)

  local g2 = fusibles.nghToNnGraph(x)
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

  local x = fusibles.nnGraphToNgh(g)
  
  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  while fusion.doFuseIteration(x) do
  end
  fusion.generateKernels(x)
  fusibles.printGraphWithLinks(x)
  tester:asserteq(x:walkValidate(), true)

  local g2 = fusibles.nghToNnGraph(x)
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

  local x = fusibles.nnGraphToNgh(g)
  
  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  while fusion.doFuseIteration(x) do
  end
  fusion.generateKernels(x)
  fusibles.printGraphWithLinks(x)
  tester:asserteq(x:walkValidate(), true)

  local g2 = fusibles.nghToNnGraph(x)
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

  x = fusibles.nnGraphToNgh(lstm)
  fusibles.printGraph(x)
  if os.getenv('TESTS') ~= nil then
    x:dot('', 'lstm1')
  end

  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(x:walkValidate(), true)
  fusion.generateKernels(x)

  local g2 = fusibles.nghToNnGraph(x)
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
--  x:dot('', 'xit' .. it)
--  while fusion.doFuseIteration(x) do
--    it = it + 1
--    print('it ' .. it .. ' ======================')
--    tester:asserteq(x:walkValidate(), true)
--    x:dot('', 'xit' .. it)
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

  x = fusibles.nnGraphToNgh(lstm)
  fusibles.walkApply(x, function(node)
    fusibles.nodeSetName(node, 'node ' .. node.id)
  end)
  fusibles.printGraph(x)
  if os.getenv('TESTS') ~= nil then x:dot('', 'lstm1') end

  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(x:walkValidate(), true)

  it = 0
  print('it ' .. it .. ' ======================')
  if os.getenv('TESTS') ~= nil then x:dot('', 'xit' .. it) end
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(x:walkValidate(), true)
    if os.getenv('TESTS') ~= nil then x:dot('', 'xit' .. it) end
  end

  fusion.generateKernels(x)

  fusibles.printGraphWithLinks(x)
  local g2 = fusibles.nghToNnGraph(x)
  if os.getenv('TESTS') ~= nil then graph.dot(g2.fg, '', 'g2') end
  local outputafter = g2:forward(inputs)
  print('outputafter', outputafter)

  for o=1, #outputbefore do
    diff = (outputafter[o] - outputbefore[o]):abs():sum()
    assert(diff == 0)
  end

--  local it = 0
--  print('it ' .. it .. ' ======================')
--  x:dot('', 'xit' .. it)
--  while fusion.doFuseIteration(x) do
--    it = it + 1
--    print('it ' .. it .. ' ======================')
--    tester:asserteq(x:walkValidate(), true)
--    x:dot('', 'xit' .. it)
--  end

--  fusion.generateKernels(x)
end

function fusiontests.testLSTM()
  require('test.lstm.OneHot')
  local LSTM = require('test.lstm.LSTM')
  lstm = LSTM.lstm(65, 128, 2, 0)
  graph.dot(lstm.fg, '', 'lstm.g')
  x = fusibles.nnGraphToNgh(lstm)
  fusibles.printGraph(x)
  x:dot('', 'lstm1')

  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(x:walkValidate(), true)

  local it = 0
  print('it ' .. it .. ' ======================')
  x:dot('', 'xit' .. it)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(x:walkValidate(), true)
    x:dot('', 'xit' .. it)
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

