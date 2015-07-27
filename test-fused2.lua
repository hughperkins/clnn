require 'nngraph'
require 'clnn'

local ngh = require('nodeGraphHelper')
local gh = require('graphHelper')
local fusion = require('fusion')

--local n = nn.Apply(3, 2, [[
--  {{out1}} = {{in1}} + {{in2}};
--  {{out2}} = {{in3}} + 3.0f;
--]], [[
--  {{in1}} = {{out1}};
--  {{in2}} = {{out1}};
--  {{in3}} = {{out2}};
--]])

local in1 = torch.ClTensor(3,2):uniform()
local in2 = torch.ClTensor(3,2):uniform()
local in3 = torch.ClTensor(3,2):uniform()
local inputs = {in1, in2, in3}

--local outputs = n:forward(inputs)
--print('in1', in1)
--print('in2', in2)
--print('in3', in3)
--print('outputs\n', outputs, outputs[1], outputs[2])

--local gradInput = n:backward(inputs, outputs)
--print('gradInput\n', gradInput, gradInput[1], gradInput[2], gradInput[3])

local x = nn.Identity()()
local n1 = nn.Tanh()(x)
local n2 = nn.Sigmoid()(n1)
local n3 = nn.Sigmoid()(x)
local n4 = nn.CAddTable()({n2, n3})

ngh.nameNode(x, 'x')
ngh.nameNode(n1, 'n1')
ngh.nameNode(n2, 'n2')
ngh.nameNode(n3, 'n3')
ngh.nameNode(n4, 'n4')

--local m3 = nn.Tanh()(m2)
g = nn.gModule({x}, {n2})
g2 = g:clone()
g:cl()
g2:cl()

local output1 = g:forward(inputs[1])
local gradInput1 = g:backward(inputs[1], output1)

if false then
  convertGraphToApply(g2)
  local output2 = g2:forward(inputs[1])
  local diff = (output2 - output1):abs():sum()
  print('diff', diff)
  if diff ~= 0 then
    print('output1', output1)
    print('output2', output2)
    assert(diff == 0)
  end

  local gradInput2 = g2:backward(inputs[1], output1)

  diff = (gradInput2 - gradInput1):abs():sum()
  print('diff', diff)
  if diff ~= 0 then
    print('gradInput1\n', gradInput1)
    assert(diff == 0)
  end
end

function isApply(module)
  if torch.type(module) == 'nn.Apply' then
    return true
  end
  return false
end

--if false then
--  nodes3 = fuseApply(nodes3)

--  graph.dot(nodes3:graph(), '', 'nodes3')

--  g3 = nn.gModule({x3}, {nodes3})

--  graph.dot(g3.fg, '', 'g3.fg')

--  print('input', inputs[1])

--  output2 = g3:forward(inputs[1])
--  print('output1', output1)
--  print('output2', output2)
--  diff = (output2 - output1):abs():sum()
--  print('diff', diff)
--  assert(diff == 0)

--  local gradInput2 = g3:backward(inputs[1], output1)
--  print('gradInput2\n', gradInput2)

--  diff = (gradInput2 - gradInput1):abs():sum()
--  print('diff', diff)
--  assert(diff == 0)
--end

x3, nodes3 = gh.graphGetNodes(g2)

ngh.walkAddParents(nodes3)
ngh.walkStripByObjects(nodes3)
--ngh.walkAddDataIds(nodes3)
--ngh.walkReverseAddDataIds(nodes3)
ngh.walkReverseAddDataIds(x3)
print('x3', ngh.nodeToString(x3))
print('nodes3', ngh.nodeToString(nodes3))
ngh.printGraph(nodes3)
ngh.reversePrintGraph(x3)
ngh.walkApply(nodes3, function(node) print(ngh.nodeToString(node) .. ' ' .. tostring(node.parents ~= nil)) end)
print('x3.parents[1]', ngh.nodeToString(x3.parents[1]))

fusion.walkConvertToApply(nodes3)
ngh.reversePrintGraph(x3)

if os.getenv('FUSE') ~= nil then
  print('fusing...')
  p, c = fusion.getFusiblePair(nodes3)
  -- p == parent, c == child
  -- fuse(p, c) = p . c = p(c(input)) 
  -- output = p(c(input))

  print('p ~= nil', parent ~= nil)
  print('p', ngh.nodeToString(p))
  print('c', ngh.nodeToString(c))

  local pmod = p.data.module
  local cmod = c.data.module

  local p_inputs = pmod.numInputs
  local c_inputs = cmod.numInputs
  local p_outputs = pmod.numOutputs
  local c_outputs = cmod.numOutputs

  local virtualOutputs = c.virtualOutputs or 0
  -- virtualOutputs = virtualOutputs + mod1.numOutputs
  -- mod1.virtualOutputs = virtualOutputs

  local cf = cmod.forwardExpression
  local pf = pmod.forwardExpression
  local cf = cf:gsub('{{output}}', 'float {{virtualOut' .. (virtualOutputs + 1) .. '}}')
  local pf = pf:gsub('{{input}}', '{{virtualOut' .. (virtualOutputs + 1) .. '}}')
  for o=1,cmod.numOutputs do
    print('o', o)
    virtualOutputs = virtualOutputs + 1
    cf = cf:gsub('{{output' .. o .. '}}', 'float {{virtualOut' .. virtualOutputs .. '}}')
    pf = pf:gsub('{{input' .. o .. '}}', '{{virtualOut' .. virtualOutputs .. '}}')
  end

  print('cf', cf)
  print('pf', pf)

  p.data.module.forwardExpression = cf .. '\n' .. pf
  --nodes3 = ngh.removeNodeByWalk(nodes3, n2.data)

--  ngh.walkApply(nodes3, function(node)
--    print(node.data.id, node.data.module, 'parentssize', #node.parents, torch.type(node.parents))
--  end)

--  print(n1.data.id, n1.data.module, 'parentssize', #n1.parents, torch.type(n1.parents))

  fused = ngh.reduceEdge(p, c)
--  fused.forwardExpression = 

  -- remove n2 from graph
  -- here we assume that n2 only has n1 as parent, and n1 only has n2 as child
  -- need to improve this later...
  --n2.parents = {} 
  --n1.children = {}
  -- add n2's children as n1's
  --for i, n2child in n2.children do
  --  
  --end
end

ngh.printGraph(nodes3)
ngh.reversePrintGraph(x3)

ngh.walkAddReciprocals(nodes3)
graph.dot(nodes3:graph():reverse(), '', 'nodes3')

