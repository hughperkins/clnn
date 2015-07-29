fusion = {}

-- this assumes inverted, wrt g.bg, ie root of nodes, highest parent, is x, input
-- and bottom, the child, is results
-- in other words, its inverted wrt the graph created by doing like nn.Tanh()(nn.Sigmoid()()),
-- which would make tanh a parent of sigmoid.  here we expect tanh to be the child of sigmoid,
-- in this example

local ngh = require('nodeGraphHelper')

function fusion.isApply(module)
  if torch.type(module) == 'nn.Apply' then
    return true
  end
  return false
end

function fusion.convertToApply(node)
  local moduletype = torch.type(node.data.module)
  if moduletype == 'nn.Tanh' then
    local dat = node.data
    dat.name = moduletype
    dat.virtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output}} = tanh({{input}});', transforms={input='input', output='output'}})
    table.insert(dat.beobj, {template='{{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});',
      transforms={gradInput='gradInput', gradOutput='gradOutput', output='output'}})
    local apply = nn.Apply(1, 1, [[
      {{output}} = tanh({{input}});
    ]], [[
      {{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});
    ]], moduletype)
    node.data.module = apply
  elseif moduletype == 'nn.Sigmoid' then
    local dat = node.data
    dat.name = moduletype
    dat.virtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output}} = 1.f / (1.f + exp( - {{input}}));', transforms={input='input', output='output'}})
    table.insert(dat.beobj, {template='{{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});',
      transforms={gradInput='gradInput', gradOutput='gradOutput', output='output'}})
    local apply = nn.Apply(1, 1, [[
      {{output}} =  1.f / (1.f + exp( - {{input}}));
    ]], [[
      {{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});
    ]], moduletype)
    node.data.module = apply
  elseif moduletype == 'nn.Exp' then
    local dat = node.data
    dat.name = moduletype
    dat.virtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output}} = exp({{input}});', transforms={input='input', output='output'}})
    table.insert(dat.beobj, {template='{{gradInput}} = {{gradOutput}} * {{output}};',
      transforms={gradInput='gradInput', gradOutput='gradOutput', output='output'}})
    local apply = nn.Apply(1, 1, [[
      {{output}} =  exp({{input}});
    ]], [[
      {{gradInput}} = {{gradOutput}} * {{output}};
    ]], moduletype)
    node.data.module = apply
  elseif moduletype == 'nn.Abs' then
    local dat = node.data
    dat.name = moduletype
    dat.virtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output}} = fabs({{input}});', transforms={input='input', output='output'}})
    table.insert(dat.beobj, {template='{{gradInput}} = {{input}} < 0 ? - {{gradOutput}} : {{gradOutput}};',
      transforms={gradInput='gradInput', gradOutput='gradOutput', input='input'}})
    local apply = nn.Apply(1, 1, [[
      {{output}} =  fabs({{input}});
    ]], [[
      {{gradInput}} = {{input}} < 0 ? - {{gradOutput}} : {{gradOutput}};
    ]], moduletype)
    node.data.module = apply
  elseif moduletype == 'nn.CAddTable' then
    local dat = node.data
    dat.name = moduletype
    dat.virtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output}} = {{input1}} + {{input2}};', transforms={input1='input1', input2='input2', output='output'}})
    table.insert(dat.beobj, {template=
[[{{gradInput1}} = {{gradOutput}};
{{gradInput2}} = {{gradOutput}};]],
      transforms={gradInput1='gradInput1', gradInput2='gradInput2', gradOutput='gradOutput'}})
    local apply = nn.Apply(2, 1, [[
      {{output}} = {{input1}} + {{input2}};
    ]], [[
      {{gradInput1}} = {{gradOutput}};
      {{gradInput2}} = {{gradOutput}};
    ]], moduletype)
    node.data.module = apply
  end
end

function fusion.walkConvertToApply(nodes)
  ngh.walkApply(nodes, function(node)
    fusion.convertToApply(node)
  end)
end

function fusion.reverseWalkConvertToApply(x)
  ngh.reverseWalkApply(x, function(node)
    fusion.convertToApply(node)
  end)
end

function fusion.getFusiblePair(x)
  local n1 = nil
  local n2 = nil
  ngh.walkApply(x, function(node)
    if n1 ~= nil then
      return
    end
    if fusion.isApply(node.data.module) then
      for j, child in ipairs(node.children) do  -- I know this is rubbish n-squared, fix this later..
        if fusion.isApply(child.data.module) then
          n1 = node
          n2 = child
          return
        end
      end
    end
  end)
  return n1, n2
end

function fusion.doFuse(x)
  while fusion.doFuseIteration(x) do
  end
end

-- since we inverted this:
-- child function is applied to result of parent function
function fusion.doFuseIteration(x)
  p, c = fusion.getFusiblePair(x)
  -- p == parent, c == child
  -- fuse(p, c) = c . p = c(p(input)) 
  -- output = c(p(input))

  if p == nil then
    return false
  end

  local pdat = p.data
  local cdat = c.data
  local pmod = pdat.module
  local cmod = cdat.module

  local p_inputs = pmod.numInputs
  local c_inputs = cmod.numInputs
  local p_outputs = pmod.numOutputs
  local c_outputs = cmod.numOutputs

--  print('parent virtualoutputs', p.data.module.virtualOutputs)
--  print('child virtualoutputs', c.data.module.virtualOutputs)
  local virtualOutputs = (c.data.module.virtualOutputs or 0) + (p.data.module.virtualOutputs or 0)
  -- TODO need to renumber either all parents or all childs virtualoutputs, so dont overlap
  print('virtualOutputs before =', virtualOutputs)
  -- virtualOutputs = virtualOutputs + mod1.numOutputs
  -- mod1.virtualOutputs = virtualOutputs

  -- observations:
  -- ALL parent's outputs go to child
  -- but normally parent will just have one unique output (could go to multiple children)
  -- child might have more than one input
  -- need to find which input comes from child
  parentIsWhichInput = ngh.getLinkPos(c.parents, p)
  print('parent input:', parentIsWhichInput)

  local pfo = pdat.feobj
  local cfo = cdat.feobj
  local fusedfo = {}
  for i, feo in ipairs(pfo) do
    print('inserting', feo)
    table.insert(fusedfo, feo)
  end
  for i, feo in ipairs(cfo) do
    print('inserting', feo)
    table.insert(fusedfo, feo)
  end

  virtualOutputs = virtualOutputs + 1
  pfo[#pfo].transforms.output = 'float virtualOutput' .. virtualOutputs
  if cfo[1].transforms.input ~= nil then
    cfo[1].transforms.input = 'virtualOutput' .. virtualOutputs
  end
  if cfo[1].transforms['input' .. parentIsWhichInput] ~= nil then
    cfo[1].transforms['input' .. parentIsWhichInput] = 'virtualOutput' .. virtualOutputs
  end
  for o=1,pmod.numOutputs do
--    cfo
--    cf = cf:gsub('{{output' .. o .. '}}', 'float {{virtualOut' .. virtualOutputs .. '}}')
--    pf = pf:gsub('{{input' .. o .. '}}', '{{virtualOut' .. virtualOutputs .. '}}')
  end

  local fused = ngh.reduceEdge(p, c)
  local fdat = fused.data
  fdat.feobj = fusedfo
  local fmod = fdat.module
  fmod.forwardExpression = fusedExp
  fmod.virtualOutputs = virtualOutputs
  ngh.nodeSetName(fused, ngh.nodeGetName(c) .. '.' .. ngh.nodeGetName(p))

  return true
end

return fusion

