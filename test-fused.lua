require 'clnn'

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
--local m2 = nn.Sigmoid()(m1)
g = nn.gModule({x}, {m1})
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
      {{gradInput}} = {{input}} * (1 - {{gradOutput}} * {{gradOutput}});
    ]])
    node.data.module = apply
    print('node.data.module', node.data.module)
  end
end

print('redo iteration ==============================')
for i, node in ipairs(g2.forwardnodes) do
  print(i, node, node.id)
  local moduletype = torch.type(node.data.module)
  print('node.data.module', node.data.module)
  if moduletype == 'nn.Tanh' then
    print('Tanh detected')
  end
end

graph.dot(g2.fg, '', 'g2.fg')

local output2 = g2:forward(inputs[1])
print('output1', output1)
print('output2', output2)
print('diff', (output2 - output1):abs():sum())

