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
local n3 = nn.Exp()(x)
--local n4 = nn.Abs()(n3)
--local n5 = nn.Sigmoid()(x)
local n4 = nn.CAddTable()({n2, n3})

ngh.nameNode(x, 'x')
ngh.nameNode(n1, 'n1')
ngh.nameNode(n2, 'n2')
ngh.nameNode(n3, 'n3')
ngh.nameNode(n4, 'n4')
--ngh.nameNode(n5, 'n5')
--ngh.nameNode(n6, 'n6')

--local m3 = nn.Tanh()(m2)
g = nn.gModule({x}, {n4})
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
ngh.walkRemoveBidirectional(nodes3)
x3 = ngh.invertGraph(nodes3)
--ngh.walkAddDataIds(nodes3)
--ngh.walkReverseAddDataIds(nodes3)
ngh.walkAddDataIds(x3)
print('x3', ngh.nodeToString(x3))
print('nodes3', ngh.nodeToString(nodes3))
ngh.printGraph(x3)
ngh.reversePrintGraph(nodes3)
print('x3.children[1]', ngh.nodeToString(x3.children[1]))

fusion.walkConvertToApply(x3)
ngh.printGraph(x3)

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


if os.getenv('FUSE') ~= nil then
  print('fusing...')
  while fusion.doFuse(nodes3) do
  end
--  doFuse(nodes3)
--  doFuse(nodes3)
end

ngh.printGraph(x3)
--ngh.reversePrintGraph(x3)

--ngh.walkAddReciprocals(nodes3)
--graph.dot(nodes3:graph():reverse(), '', 'nodes3')

print('x3')
ngh.printGraph(x3)
ngh.dot(x3, '', 'x3')

