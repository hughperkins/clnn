require 'nngraph'
require 'clnn'

--local fusibles = require('Fusible')
local gh = require('graphHelper')
local fusion = require('fusion')
local Fusible = nn.Fusible

local fusiontests = {}

--function nngraph.Node:graphNodeName()
--  if self.id ~= nil then
--    res = tostring(self.id)
--    local dat = self
--    if dat.module ~= nil then
--      local mod = dat.module
--      res = res .. ' ' .. torch.type(mod)
--      if mod.numInputs ~= nil then
--        if dat.numVirtualOutputs ~= nil and dat.numVirtualOutputs > 0 then
--          res = res .. ' ' .. mod.numInputs.. ' -> (' .. dat.numVirtualOutputs .. ')' .. ' -> ' .. mod.numOutputs
--        else
--          res = res .. ' ' .. mod.numInputs .. ' -> ' .. mod.numOutputs
--        end
--      end
--    end
--    return res
--  end
--  if self.annotations.name then
--    return self.annotations.name .. ' (' .. self.id .. ')'
--  else
--    return 'Node' .. self.id
--  end
--end

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
  tester:asserteq(x.outputs[1].child.numVirtualOutputs, 1)
  n1 = x.outputs[1].child
  tester:asserteq(#n1.feobj, 1)
--  tester:asserteq(#n1.beobj, 1)
  tester:asserteq(n1.feobj[1].template, '{{output1}} = tanh({{input1}});')
--  tester:asserteq(n1.beobj[1].template, '{{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});')
  tester:asserteq(n1.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(n1.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(n1.feobj[1].transforms.input1.inputIdx, 1)
  tester:asserteq(n1.feobj[1].transforms.output1.outputIdx, 1)

  fusion.generateKernels(x)
end

function fusiontests.testApplyConvertSigmoid()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)

  x = nn.Fusible.fromNodes(n1)

  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(x:walkValidate(), true)
  n1 = x:firstChild()

  tester:asserteq(torch.type(n1.module), 'nn.Apply')
  tester:asserteq(torch.type(x.module), 'nn.Identity')
  tester:asserteq(n1.numVirtualOutputs, 1)
  tester:asserteq(#n1.feobj, 1)
--  tester:asserteq(#n1.beobj, 1)
  tester:asserteq(n1.feobj[1].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
--  tester:asserteq(n1.beobj[1].template, '{{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});')
  tester:asserteq(n1.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(n1.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(n1.feobj[1].transforms.input1.inputIdx, 1)
  tester:asserteq(n1.feobj[1].transforms.output1.outputIdx, 1)

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
  tester:asserteq(n2.numVirtualOutputs, 1)
  tester:asserteq(#n2.feobj, 1)
--  tester:asserteq(#n2.beobj, 1)
  tester:asserteq(n2.feobj[1].template, '{{output1}} = tanh({{input1}});')
--  tester:asserteq(n2.beobj[1].template, '{{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});')
  tester:asserteq(n2.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(n2.feobj[1].transforms.input1.inputIdx, 1)
  tester:asserteq(n2.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(n2.feobj[1].transforms.output1.outputIdx, 1)

  tester:asserteq(torch.type(n1.module), 'nn.Apply')
  tester:asserteq(n1.numVirtualOutputs, 1)
  tester:asserteq(#n1.feobj, 1)
--  tester:asserteq(#n1.beobj, 1)
  tester:asserteq(n1.feobj[1].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
--  tester:asserteq(n1.beobj[1].template, '{{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});')
  tester:asserteq(n1.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(n1.feobj[1].transforms.input1.inputIdx, 1)
  tester:asserteq(n1.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(n1.feobj[1].transforms.output1.outputIdx, 1)

  tester:asserteq(#n1.outputs, 1)
  tester:asserteq(n1.outputs[1].child, n2)
  tester:asserteq(n1.outputs[1].outputIdx, 1)
  tester:asserteq(n1.outputs[1].inputIdx, 1)

--  fusion.generateKernels(x)
end

function fusiontests.testOutputsTwoOutput()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.Exp()(n1)
  local n3 = nn.Tanh()(n1)
  local out = nn.Identity()({n2, n3})

  x.data.annotations.name = 'x'
  n1.data.annotations.name = 'n1'
  n2.data.annotations.name = 'n2'
  n3.data.annotations.name = 'n3'
  out.data.annotations.name = 'out'

  x = nn.Fusible.fromNodes(out)
  n1 = x:firstChild()
  n2 = n1:firstChild()
  n3 = n1.outputs[2].child
  out = n2:firstChild()

  tester:asserteq(x:walkValidate(), true)
  fusion.walkConvertToApply(x)
  tester:asserteq(x:walkValidate(), true)
  if os.getenv('TESTS') ~= nil then x:dot('', 'x') end

  tester:asserteq(#x.outputs, 1) 
  tester:asserteq(#n1.outputs, 2) 
  tester:asserteq(#n2.outputs, 1) 
  tester:asserteq(#n3.outputs, 1) 
  tester:asserteq(#out.outputs, 0) 
  tester:asserteq(n1.outputs[1].outputIdx, 1) 
  tester:asserteq(n1.outputs[1].child, n2)
  tester:asserteq(n1.outputs[2].outputIdx, 2) 
  tester:asserteq(n1.outputs[2].child, n3)
end

function fusiontests.testFuseTanhSigmoid()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.Tanh()(n1)
  local out = nn.Identity()({n2})

  x.data.annotations.name = 'x'
  n1.data.annotations.name = 'n1'
  n2.data.annotations.name = 'n2'
  out.data.annotations.name = 'out'

  x = nn.Fusible.fromNodes(out)
  n1 = x:firstChild()
  n2 = n1:firstChild()
  out = n2:firstChild()

  tester:asserteq(x:walkValidate(), true)

  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 4)
  if os.getenv('TESTS') ~= nil then x:dot('', 'xbefore') end
  tester:asserteq(x:walkValidate(), true)
  fusion.walkAssignVirtualIdx(x)
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
  tester:asserteq(fdat.feobj[1].transforms.input1.inputIdx, 1)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[1].transforms.output1.outputIdx, nil)
  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[2].transforms.input1.inputIdx, nil)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[2].transforms.output1.outputIdx, 1)

  tester:asserteq(#fused.outputs, 1)
  print('x.child.out[1]', fused.outputs[1].child.module)
  print('out', out.module)
  tester:asserteq(fused.outputs[1].child, out)
  tester:asserteq(fused.outputs[1].outputIdx, 1)

  for i, feobj in ipairs(fdat.feobj) do
    for k, v in pairs(feobj.transforms) do
      print('feobj[' .. i .. ']', k, v)
    end
  end

  fusion.generateKernels(x)
end

function fusiontests.testFuseExpTanhSigmoid()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.Tanh()(n1)
  local n3 = nn.Exp()(n2)
  local out = nn.Identity()({n3})

  x.data.annotations.name = 'x'
  n1.data.annotations.name = 'n1'
  n2.data.annotations.name = 'n2'
  n3.data.annotations.name = 'n3'
  out.data.annotations.name = 'out'

  x = nn.Fusible.fromNodes(out)

  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 5)
  fusion.walkAssignVirtualIdx(x)
  fusion.doFuse(x)
  tester:asserteq(x:walkValidate(), true)
  tester:asserteq(x:count(), 3)

  tester:asserteq(torch.type(x.module), 'nn.Identity')

  tester:asserteq(x.name, 'x')
  tester:asserteq(#x.outputs, 1)
  tester:asserteq(#x.outputs[1].child.outputs, 1)

  local fused = x.outputs[1].child
  local fdat = fused
  tester:asserteq(fused.name, 'n3.n2.n1')
  tester:asserteq(#fdat.feobj, 3)
  tester:asserteq(fdat.feobj[1].template, '{{output1}} = 1.f / (1.f + exp( - {{input1}}));')
  tester:asserteq(fdat.feobj[2].template, '{{output1}} = tanh({{input1}});')
  tester:asserteq(fdat.feobj[3].template, '{{output1}} = exp({{input1}});')

  tester:asserteq(fdat.feobj[1].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.input1.inputIdx, 1)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[1].transforms.output1.virtualIdx, 1)
  tester:asserteq(fdat.feobj[1].transforms.output1.outputIdx, nil)

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[2].transforms.input1.inputIdx, nil)
  tester:asserteq(fdat.feobj[2].transforms.input1.virtualIdx, 1)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[2].transforms.output1.outputIdx, nil)
  tester:asserteq(fdat.feobj[2].transforms.output1.virtualIdx, 2)

  tester:asserteq(fdat.feobj[3].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[3].transforms.input1.inputIdx, nil)
  tester:asserteq(fdat.feobj[3].transforms.input1.virtualIdx, 2)
  tester:asserteq(fdat.feobj[3].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[3].transforms.output1.outputIdx, 1)
  tester:asserteq(fdat.feobj[3].transforms.output1.virtualIdx, 3)

  for i, feobj in ipairs(fdat.feobj) do
    for k, v in pairs(feobj.transforms) do
      print('feobj[' .. i .. ']', k, v)
    end
  end

  fusion.generateKernels(x)
end

function fusiontests.testApplyConvertSigmoidAddTable()
  local x = nn.Identity()()
  local n1 = nn.Sigmoid()(x)
  local n2 = nn.CAddTable()({n1, x})
  local out = nn.Identity()({n2})

  x.data.annotations.name = 'x'
  n1.data.annotations.name = 'n1'
  n2.data.annotations.name = 'n2'
  out.data.annotations.name = 'out'

  if os.getenv('TESTS') ~= nil then graph.dot(out:graph(), '', 'out.graph') end
  x = nn.Fusible.fromNodes(out)
  if os.getenv('TESTS') ~= nil then x:printGraphWithLinks() end

  fusion.walkConvertToApply(x)
  fusion.walkAssignVirtualIdx(x)
  if os.getenv('TESTS') ~= nil then x:printGraphWithLinks() end
  tester:asserteq(x:count(), 4)
  tester:asserteq(x:walkValidate(), true)
  if os.getenv('TESTS') ~= nil then x:dot('', 'xbefore') end
  fusion.doFuse(x)
  if os.getenv('TESTS') ~= nil then x:printGraphWithLinks() end
  if os.getenv('TESTS') ~= nil then x:dot('', 'xafter') end
  tester:asserteq(x:walkValidate(), true)
  tester:asserteq(x:count(), 3)

  tester:asserteq(torch.type(x.module), 'nn.Identity')

  local fused = x:firstChild()
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
  tester:asserteq(fdat.feobj[1].transforms.input1.inputIdx, 1)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[1].transforms.output1.outputIdx, 1)

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input1.inputIdx, 1)
  tester:asserteq(fdat.feobj[2].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[2].transforms.input2.idx, 2)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[2].transforms.output1.outputIdx, 1)

  fusion.generateKernels(x)
end

function fusiontests.testApplyConvertMultiInputAdd()
  local x = nn.Identity()()
  local x1, x2 = x:split(2)
  local n1 = nn.Tanh()(x1)
  local n2 = nn.CAddTable()({n1, x2})

  x.data.annotations.name = 'x'
  x1.data.annotations.name = 'x1'
  x2.data.annotations.name = 'x2'
  n1.data.annotations.name = 'n1'
  n2.data.annotations.name = 'n2'

  x = nn.Fusible.fromNodes(n2)
  x1 = x:firstChild():firstChild()
  x2 = x:firstChild().outputs[2].child
  n1 = x1:firstChild()
  n2 = n1:firstChild()

  fusion.walkConvertToApply(x)
--  x:dot('', 'add')
  tester:asserteq(x:count(), 6)
  tester:asserteq(x:walkValidate(), true)
--  x:dot('', 'xold')
  tester:asserteq(x:walkValidate(), true)
--  local xold = x:walkClone()
--  tester:asserteq(x:walkValidate(), true)
--  x:printGraph()
  fusion.doFuse(x)
  tester:asserteq(x:walkValidate(), true)
--  x:printGraph()
--  x:dot('', 'xnew')
  tester:asserteq(x:count(), 5)

  tester:asserteq(torch.type(x.module), 'nn.Identity')

  local fused = x:firstChild():firstChild():firstChild()
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
  tester:asserteq(fdat.feobj[1].transforms.input1.inputIdx, 1)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[1].transforms.output1.outputIdx, 1)

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input1.inputIdx, 1)
  tester:asserteq(fdat.feobj[2].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[2].transforms.input2.idx, 2)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[2].transforms.output1.outputIdx, 1)

  tester:asserteq(Fusible.getLinkPos(x1:firstChild().inputs, x1), 1)
  tester:asserteq(Fusible.getLinkPos(x2:firstChild().inputs, x2), 2)

  fusion.generateKernels(x)
end

function fusiontests.testApplyConvertMultiInputAdd3()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.CMulTable()({x1, x2})
  local n2 = nn.CAddTable()({n1, x3})

  x.data.annotations.name = 'x'
  x1.data.annotations.name = 'x1'
  x2.data.annotations.name = 'x2'
  x3.data.annotations.name = 'x3'
  n1.data.annotations.name = 'n1'
  n2.data.annotations.name = 'n2'

  x = nn.Fusible.fromNodes(n2)
  x1 = x:firstChild():firstChild()
  x2 = x:firstChild().outputs[2].child
  x3 = x:firstChild().outputs[3].child
  n1 = x1:firstChild()
  n2 = n1:firstChild()

  fusion.walkConvertToApply(x)
  x:dot('', 'add')
  tester:asserteq(x:count(), 7)
  tester:asserteq(x:walkValidate(), true)
  x:dot('', 'xold')
  tester:asserteq(x:walkValidate(), true)
  local xold = x:walkClone()
  tester:asserteq(x:walkValidate(), true)
--  x:printGraph()
  fusion.doFuse(x)
  tester:asserteq(x:walkValidate(), true)
--  x:printGraph()
  x:dot('', 'xnew')
  tester:asserteq(x:count(), 6)

  tester:asserteq(torch.type(x.module), 'nn.Identity')

  local fused = x:firstChild():firstChild():firstChild()
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
  tester:asserteq(fdat.feobj[1].transforms.input1.inputIdx, 1)
  tester:asserteq(fdat.feobj[1].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.input2.idx, 2)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[1].transforms.output1.outputIdx, 1)

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input1.inputIdx, 1)
  tester:asserteq(fdat.feobj[2].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[2].transforms.input2.idx, 3)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[2].transforms.output1.outputIdx, 1)

  tester:asserteq(Fusible.getLinkPos(x1:firstChild().inputs, x1), 1)
  tester:asserteq(Fusible.getLinkPos(x2:firstChild().inputs, x2), 2)
  tester:asserteq(Fusible.getLinkPos(x3:firstChild().inputs, x3), 3)

  fusion.generateKernels(x)
end

function fusiontests.testAddTanhMul()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.CAddTable()({x1, x2})
  local n2 = nn.Tanh()({n1})
  local n3 = nn.CMulTable()({n2, x3})

  x.data.annotations.name = 'x'
  x1.data.annotations.name = 'x1'
  x2.data.annotations.name = 'x2'
  x3.data.annotations.name = 'x3'

  n1.data.annotations.name = 'n1'
  n2.data.annotations.name = 'n2'
  n3.data.annotations.name = 'n3'

  tester:asserteq(n1.data.annotations.name, 'n1')

  if os.getenv('TESTS') ~= nil then graph.dot(n3:graph(), '', 'n3g') end

  tester:asserteq(n1.data.annotations.name, 'n1')

  x = nn.Fusible.fromNodes(n3)
  x1 = x:firstChild().outputs[1].child
  x2 = x:firstChild().outputs[2].child
  x3 = x:firstChild().outputs[3].child
  n1 = x1:firstChild()
  n2 = n1:firstChild()
  n3 = n2:firstChild()

  tester:asserteq(x.name, 'x')
  tester:asserteq(n1.name, 'n1')
  tester:asserteq(n2.name, 'n2')

  if os.getenv('TESTS') ~= nil then x:dot('', 'testAddTanhMulBefore') end
  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 8)
  tester:asserteq(x:walkValidate(), true)
  x:printGraph()

  tester:asserteq(n1.name, 'n1')

  local it = 0
  print('it ' .. it .. ' ===============')
  x:printGraphWithLinks()
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ===============')
    tester:asserteq(x:walkValidate(), true)
    x:printGraphWithLinks()
  end

--  x:printGraphWithLinks()

  tester:asserteq(x:walkValidate(), true)
  x:printGraph()
  if os.getenv('TESTS') ~= nil then x:dot('', 'testAddTanhMulAfter') end
  tester:asserteq(x:count(), 6)

  local fused = x:firstChild():firstChild():firstChild()
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
  tester:asserteq(fdat.feobj[1].transforms.input1.inputIdx, 1)
  tester:asserteq(fdat.feobj[1].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[1].transforms.input2.idx, 2)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[1].transforms.output1.outputIdx, 1)

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input1.inputIdx, 1)
--  tester:asserteq(fdat.feobj[2].transforms.input2.src, 'input')
--  tester:asserteq(fdat.feobj[2].transforms.input2.idx, 3)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.output1.outputIdx, 2)

  tester:asserteq(fdat.feobj[3].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[3].transforms.input1.inputIdx, 2)
  tester:asserteq(fdat.feobj[3].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[3].transforms.input2.idx, 3)
  tester:asserteq(fdat.feobj[3].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[3].transforms.output1.outputIdx, 1)

  tester:asserteq(Fusible.getLinkPos(x1:firstChild().inputs, x1), 1)
  tester:asserteq(Fusible.getLinkPos(x2:firstChild().inputs, x2), 2)
  tester:asserteq(Fusible.getLinkPos(x3:firstChild().inputs, x3), 3)

  fusion.generateKernels(x)
end

function fusiontests.testSigMulAdd()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.Sigmoid()({x2})
  local n2 = nn.CMulTable()({x1, n1})
  local n3 = nn.CAddTable()({n2, x3})

  x.data.annotations.name = 'x'
  x1.data.annotations.name = 'x1'
  x2.data.annotations.name = 'x2'
  x3.data.annotations.name = 'x3'

  n1.data.annotations.name = 'n1'
  n2.data.annotations.name = 'n2'
  n3.data.annotations.name = 'n3'

  x = nn.Fusible.fromNodes(n3)
  x1 = x:firstChild():firstChild()
  x2 = x:firstChild().outputs[2].child
  x3 = x:firstChild().outputs[3].child
  n1 = x2:firstChild()
  n2 = x1:firstChild()
  n3 = n2:firstChild()

  x:dot('', 'testSigMulAddBefore')
  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 8)
  tester:asserteq(x:walkValidate(), true)
  x:printGraph()

  local it = 0
  print('it ' .. it .. ' ===============')
  x:printGraphWithLinks()
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ===============')
    tester:asserteq(x:walkValidate(), true)
    x:printGraphWithLinks()
  end

--  x:printGraphWithLinks()

  tester:asserteq(x:walkValidate(), true)
  x:printGraph()
  x:dot('', 'testSigMulAddAfter')
  tester:asserteq(x:count(), 6)

  local fused = x:firstChild():firstChild():firstChild()
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
  tester:asserteq(fdat.feobj[1].transforms.input1.inputIdx, 2)
  tester:asserteq(fdat.feobj[1].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[1].transforms.output1.outputIdx, 2)

  tester:asserteq(fdat.feobj[2].transforms.input1.src, 'input')
  tester:asserteq(fdat.feobj[2].transforms.input1.inputIdx, 1)
  tester:asserteq(fdat.feobj[2].transforms.input2.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.input2.idx, 2)
  tester:asserteq(fdat.feobj[2].transforms.output1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[2].transforms.output1.outputIdx, 1)

  tester:asserteq(fdat.feobj[3].transforms.input1.src, 'virtualOutput')
  tester:asserteq(fdat.feobj[3].transforms.input1.inputIdx, 1)
  tester:asserteq(fdat.feobj[3].transforms.input2.src, 'input')
  tester:asserteq(fdat.feobj[3].transforms.input2.idx, 3)
  tester:asserteq(fdat.feobj[3].transforms.output1.src, 'output')
  tester:asserteq(fdat.feobj[3].transforms.output1.outputIdx, 1)

  tester:asserteq(Fusible.getLinkPos(x1:firstChild().inputs, x1), 1)
  tester:asserteq(Fusible.getLinkPos(x2:firstChild().inputs, x2), 2)
  tester:asserteq(Fusible.getLinkPos(x3:firstChild().inputs, x3), 3)

  fusion.generateKernels(x)
end

function fusiontests.testInputOrderThreeWay()
  local x = nn.Identity()()
  local x1, x2, x3 = x:split(3)
  local n1 = nn.Tanh()({x2})
  local n2 = nn.CAddTable()({x1, n1})
  local n3 = nn.CMulTable()({n2, x3})

  x.data.annotations.name = 'x'
  x1.data.annotations.name = 'x1'
  x2.data.annotations.name = 'x2'
  x3.data.annotations.name = 'x3'

  n1.data.annotations.name = 'n1'
  n2.data.annotations.name = 'n2'
  n3.data.annotations.name = 'n3'

  x = nn.Fusible.fromNodes(n3)
  x1 = x:firstChild():firstChild()
  x2 = x:firstChild().outputs[2].child
  x3 = x:firstChild().outputs[3].child
  n1 = x2:firstChild()
  n2 = x1:firstChild()
  n3 = n2:firstChild()

  x:dot('', 'testInputOrderThreeWayadd')
  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 8)
  tester:asserteq(x:walkValidate(), true)
  x:dot('', 'testInputOrderThreeWayBefore')
  tester:asserteq(x:walkValidate(), true)
  local xold = x:walkClone()
  tester:asserteq(x:walkValidate(), true)
  x:printGraph()
  local it = 0
  x:dot('', 'xit' .. it)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(x:walkValidate(), true)
    x:dot('', 'xit' .. it)
    fusion.generateKernels(x)
    x:printGraphWithLinks()
    x:walkApply(function(node)
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

  tester:asserteq(Fusible.getLinkPos(x1:firstChild().inputs, x1), 1)
  tester:asserteq(Fusible.getLinkPos(x2:firstChild().inputs, x2), 2)
  tester:asserteq(Fusible.getLinkPos(x3:firstChild().inputs, x3), 3)

  fusion.generateKernels(x)
end

function fusiontests.testClonedOutput()
  local name = 'testClonedOutput'

  local x = nn.Identity()()
  local n1 = nn.Tanh()(x)
  local n2 = nn.Abs()(n1)
  local n3 = nn.Sigmoid()(n1)
  local n4 = nn.CAddTable()({n2, n3})

  x.data.annotations.name = 'x'

  n1.data.annotations.name = 'n1'
  n2.data.annotations.name = 'n2'
  n3.data.annotations.name = 'n3'
  n4.data.annotations.name = 'n4'

  x = nn.Fusible.fromNodes(n4)
  n1 = x:firstChild()
  n2 = n1:firstChild()
  n3 = n1.outputs[2].child
  n4 = n2:firstChild()

  if os.getenv('TESTS') ~= nil then x:dot('', name .. 'Orig') end
  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 5)
  tester:asserteq(x:walkValidate(), true)
  if os.getenv('TESTS') ~= nil then x:dot('', name .. 'Before') end
  tester:asserteq(x:walkValidate(), true)
  local xold = x:walkClone()
  tester:asserteq(x:walkValidate(), true)
  x:printGraph()
  local it = 0
  if os.getenv('TESTS') ~= nil then x:dot('', 'xit' .. it) end
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(x:walkValidate(), true)
    if os.getenv('TESTS') ~= nil then x:dot('', 'xit' .. it) end
--    fusion.generateKernels(x)
    x:printGraphWithLinks()
    x:walkApply(function(node)
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

  tester:asserteq(x:firstChild().numInputs, 1)

--  tester:asserteq(Fusible.getLinkPos(x1:firstChild().inputs, x1), 1)
--  tester:asserteq(Fusible.getLinkPos(x2:firstChild().inputs, x2), 2)
--  tester:asserteq(Fusible.getLinkPos(x3:firstChild().inputs, x3), 3)

--  fusion.generateKernels(x)
end

function fusiontests.testFusionFromAbove()
  local x = nn.Identity()()
  local xpre, n1, n5 = x:split(3)
  local n4 = nn.CMulTable()({xpre, n1})
  local n6 = nn.CAddTable()({n4, n5})
  local n7 = nn.Tanh()(n6)
  local out = nn.Identity()({n6, n7})

  x.data.annotations.name = 'x'
  xpre.data.annotations.name = 'xpre'
  n4.data.annotations.name = 'n4'
  n6.data.annotations.name = 'n6'
  n7.data.annotations.name = 'n7'
  out.data.annotations.name = 'out'

  if os.getenv('TESTS') ~= nil then graph.dot(out:graph(), '', 'nodesbefore') end
  x = nn.Fusible.fromNodes(out)
  if os.getenv('TESTS') ~= nil then x:dot('', 'xbeforeapply') end

  x:dot('', 'add')
  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 9)
  tester:asserteq(x:walkValidate(), true)
  x:dot('', 'xold')
  tester:asserteq(x:walkValidate(), true)
  local xold = x:walkClone()
  tester:asserteq(x:walkValidate(), true)
  x:printGraph()
  local it = 0
  x:dot('', 'xit' .. it)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(x:walkValidate(), true)
    x:dot('', 'xit' .. it)
    x:printGraphWithLinks()
    x:walkApply(function(fusible)
      if fusible.feobj ~= nil then
        for i, feobj in ipairs(fusible.feobj) do
          for k, v in pairs(feobj.transforms) do
            print(sys.COLORS.Blue .. 'feobj' .. sys.COLORS.none .. '[' .. i .. ']', sys.COLORS.Yellow .. k, sys.COLORS.Yellow .. v.src .. sys.COLORS.Blue .. v.idx)
          end
          print('')
        end
      end
    end)
    fusion.generateKernels(x)
    if it >= 2 then
--      os.exit(0)
    end
  end
  tester:asserteq(x:walkValidate(), true)
  x:printGraph()
  x:dot('', 'xnew')
  tester:asserteq(x:count(), 7)
  local fused = x:firstChild():firstChild():firstChild()
  tester:asserteq(fused.numOutputs, 2)
  tester:asserteq(fused.feobj[2].output1.src, 'output')
  tester:asserteq(fused.feobj[2].output1.outputIdx, 1)
  tester:asserteq(fused.feobj[3].output1.src, 'output')
  tester:asserteq(fused.feobj[3].output1.outputIdx, 2)

  tester:asserteq(torch.type(x.module), 'nn.Identity')
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

  x.data.annotations.name = 'x'
  xpre.data.annotations.name = 'xpre'
  x1.data.annotations.name = 'x1'
  x2.data.annotations.name = 'x2'
  x3.data.annotations.name = 'x3'
  x4.data.annotations.name = 'x4'

  n1.data.annotations.name = 'n1'
  n2.data.annotations.name = 'n2'
  n3.data.annotations.name = 'n3'
  n4.data.annotations.name = 'n4'
  n5.data.annotations.name = 'n5'
  n6.data.annotations.name = 'n6'
  n7.data.annotations.name = 'n7'
  n8.data.annotations.name = 'n8'
  n9.data.annotations.name = 'n9'

  out.data.annotations.name = 'out'

  if os.getenv('TESTS') ~= nil then graph.dot(out:graph(), '', 'nodesbefore') end
  x = nn.Fusible.fromNodes(out)
  if os.getenv('TESTS') ~= nil then x:dot('', 'xbeforeapply') end

  x:dot('', 'add')
  fusion.walkConvertToApply(x)
  tester:asserteq(x:count(), 17)
  tester:asserteq(x:walkValidate(), true)
  x:dot('', 'xold')
  tester:asserteq(x:walkValidate(), true)
  local xold = x:walkClone()
  tester:asserteq(x:walkValidate(), true)
  x:printGraph()
  local it = 0
  x:dot('', 'xit' .. it)
  while fusion.doFuseIteration(x) do
    it = it + 1
    print('it ' .. it .. ' ======================')
    tester:asserteq(x:walkValidate(), true)
    x:dot('', 'xit' .. it)
--    fusion.generateKernels(x)
    x:printGraphWithLinks()
    fusion.generateKernels(x)
    if it >= 2 then
      os.exit(0)
    end
--    x:walkApply(function(node)
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
  x:printGraph()
  x:dot('', 'xnew')
  tester:asserteq(x:count(), 9)

  tester:asserteq(torch.type(x.module), 'nn.Identity')

  local fused = x:firstChild():firstChild():firstChild()
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

  local x = nn.Fusible.fromNnGraph(g)
  
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
  x:printGraphWithLinks()

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
  x:printGraphWithLinks()
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
  x:printGraphWithLinks()
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
  x:printGraphWithLinks()
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
  x:printGraph()
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
  x:walkApply(function(node)
    nn.Fusible.nodeSetName(node, 'node ' .. node.id)
  end)
  x:printGraph()
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

  x:printGraphWithLinks()
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
  x = nn.Fusible.fromNnGraph(lstm)
--  x = fusibles.nnGraphToNgh(lstm)
  x:printGraph()
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
      if k == filter or (false and k:find(filter) ~= nil) then
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

