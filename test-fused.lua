require 'clnn'

function graphGetNodes(g)
--  g2 = g:clone()
  g2 = g
  newbg = g2.bg.nodes[2]
  thisnode = newbg
  x = newbg
  while #thisnode.children > 0 do
    x = thisnode
    thisnode = thisnode.children[1]
  end

  thisnode = newbg
  while thisnode.data ~= x.data do
    thisnode = thisnode.children[1]
  end
  thisnode.children = {}

  for i, node in ipairs(newbg:graph().nodes) do
    node.data.mapindex = {}
    print('node i', i, node.data.module)
    for j, child in ipairs(node.children) do
      node.data.mapindex[#node.data.mapindex + 1] = child.data
      node.data.mapindex[child.data] = #node.data.mapindex
    end
    for k,v in pairs(node.data.mapindex) do
      print('k', torch.type(k), 'v', torch.type(v))
    end
  end

  return x, newbg
end

function removeNodeByWalk(node, data)
  print('removeNodeByWalk', node.data.annotations.name)
  if node.data == data then
    -- its me!
    assert(#node.children == 1)
    return node.children[1]
  end
  for i, child in ipairs(node.children) do
    if child.data == data then
      print('remove child', i, child.data.annotations)
      table.remove(node.children, i)
      node.children[child] = nil
      local childmapindexidx = node.data.mapindex[child.data]
      node.data.mapindex[childmapindexidx] = nil
      node.data.mapindex[child.data] = nil
      for j, childchild in ipairs(child.children) do
        if node.children[childchild] == nil then
          table.insert(node.children, childchild)
          node.children[childchild] = #node.children
          node.data.mapindex[childchild.data] = #node.data.mapindex + 1
          node.data.mapindex[#node.data.mapindex + 1] = childchild.data
        end
      end
    end
  end
  for i, child in ipairs(node.children) do
    removeNodeByWalk(child, data)
  end
  return node
end

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


require 'nngraph'
local x = nn.Identity()()
local m1 = nn.Tanh()(x)
local m2 = nn.Sigmoid()(m1)
g = nn.gModule({x}, {m2})
g2 = g:clone()
g:cl()
g2:cl()

local output1 = g:forward(inputs[1])

graph.dot(g.fg, '', 'g.fg')

for i, node in ipairs(g2.forwardnodes) do
  print(i, node, node.id)
  local moduletype = torch.type(node.data.module)
  if moduletype == 'nn.Tanh' then
    print('Tanh detected')
    local apply = nn.Apply(1, 1, [[
      {{output}} = tanh({{input}});
    ]], [[
      {{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});
    ]])
    node.data.module = apply
  elseif moduletype == 'nn.Sigmoid' then
    print('Sigmoid detected')
    local apply = nn.Apply(1, 1, [[
      {{output}} =  1.f / (1.f + exp( - {{input}}));
    ]], [[
      {{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});
    ]])
    node.data.module = apply
    print('node.data.module', node.data.module)
  end
end

graph.dot(g2.fg, '', 'g2.fg')

local output2 = g2:forward(inputs[1])
print('output1', output1)
print('output2', output2)
local diff = (output2 - output1):abs():sum()
print('diff', diff)
assert(diff == 0)

local gradInput1 = g:backward(inputs[1], output1)
print('gradInput1\n', gradInput1)

local gradInput2 = g2:backward(inputs[1], output1)
print('gradInput2\n', gradInput2)

diff = (gradInput2 - gradInput1):abs():sum()
print('diff', diff)
assert(diff == 0)


x3, nodes3 = graphGetNodes(g2)
ng3bg = nodes3:graph() -- not an nngraph, just a normal graph
ng3fg = ng3bg:reverse()
-- fuse them...
-- lets assume we've already done the search, and determined that the sigmoid and tanh are adjacent, and can be fused
-- and we know their sequence (these are both just standard graph-search type algos; I'm sure I can do this bit)
-- mind you, we need to actually get their refs / node ids here
local n1 = nil
local n2 = nil -- this is the second, that eats the first
local n1_pos = nil
local n2_pos = nil
for i, node in ipairs(ng3fg.nodes) do
  print('i', i, node.data.module)
  if node.data.module ~= nil and torch.type(node.data.module) ~= 'nn.Identity' then
    if n1 == nil then
      n1 = node
      n1_pos = i
      print('n1 is', n1.data.module)
    elseif n2 == nil then
      n2 = node
      n2_pos = i
      print('n2 is', n2.data.module)
    else
      error('shouldnt be here')
    end
  end
end

local n1_forward = n1.data.module.forwardExpression
local n2_forward = n2.data.module.forwardExpression
print('n1 forward', n1_forward)
print('n2 forward', n2_forward)

tempvar = 't1'
n1_forward = n1_forward:gsub('{{output}}', 'float ' .. tempvar)
n2_forward = n2_forward:gsub('{{input}}', tempvar)

print('n1 forward', n1_forward)
print('n2 forward', n2_forward)

local fused_forward_exp = n1_forward .. '\n' .. n2_forward
print('fused_forward', fused_forward_exp)

local fusedModule = nn.Apply(1, 1, fused_forward_exp, '')
nodes3 = removeNodeByWalk(nodes3, n2.data)
n1.data.module = fusedModule

graph.dot(nodes3:graph(), '', 'nodes3')

g3 = nn.gModule({x3}, {nodes3})

graph.dot(g3.fg, '', 'g3.fg')

print('input', inputs[1])

output2 = g3:forward(inputs[1])
print('output1', output1)
print('output2', output2)
diff = (output2 - output1):abs():sum()
print('diff', diff)
assert(diff == 0)

