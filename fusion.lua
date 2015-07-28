fusion = {}

local ngh = require('nodeGraphHelper')

function fusion.isApply(module)
  if torch.type(module) == 'nn.Apply' then
    return true
  end
  return false
end

function fusion.walkConvertToApply(nodes)
  ngh.walkApply(nodes, function(node)
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
  end)
end

function fusion.getFusiblePair(nodes)
  local n1 = nil
  local n2 = nil
  ngh.walkApply(nodes, function(node)
    if n1 ~= nil then
      return
    end
    print('fusion.getFusiblePair', node.data.id)
    if fusion.isApply(node.data.module) then
      print('  .. isApply')
      for j, child in ipairs(node.children) do  -- I know this is rubbish n-squared, fix this later..
        if fusion.isApply(child.data.module) then
          print('      .. child isApply')
          n1 = node
          n2 = child
          return
        end
      end
    end
  end)
  return n1, n2
end

function fusion.doFuse(nodes3)
  p, c = fusion.getFusiblePair(nodes3)
  -- p == parent, c == child
  -- fuse(p, c) = p . c = p(c(input)) 
  -- output = p(c(input))

  if p == nil then
    return false
  end
  print('p ~= nil', parent ~= nil)
  print('p', ngh.nodeToString(p))
  print('c', ngh.nodeToString(c))

  local pmod = p.data.module
  local cmod = c.data.module

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
  -- ALL child's outputs go to parent
  -- but normally child will just have one output
  -- parent might have more than one input
  -- only first input comes from child

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

  local fusedExp = normalizeWhite(cf .. '\n' .. pf)

  print('fusedExp', fusedExp)
--  p.data.module.forwardExpression = fusedExp
--  p.data.module.virtualOutputs = virtualOutputs

  --nodes3 = ngh.removeNodeByWalk(nodes3, n2.data)

--  ngh.walkApply(nodes3, function(node)
--    print(node.data.id, node.data.module, 'parentssize', #node.parents, torch.type(node.parents))
--  end)

--  print(n1.data.id, n1.data.module, 'parentssize', #n1.parents, torch.type(n1.parents))

  local feobj = {template='{{output}} = {{input}} * {{input}};', transforms={input='input', output='output'}}

  local cfeobj = {template='{{output}} = {{input}} * {{input}};', transforms={input='input', output='output'}}
  local pfeobj = {template='{{output}} = {{input}} * {{input}};', transforms={input='input', output='output'}}

  local cfeobj = {template='{{output}} = {{input}} * {{input}};', name='square', transforms={input='input', output='float virtualOutput1'}}
  local pfeobj = {template='{{output}} = tanh({{input}});', name='tanh', transforms={input='virtualOutput1', output='output'}}

  local fused = ngh.reduceEdge(p, c)
  local fdat = fused.data
  local fmod = fdat.module
  fmod.forwardExpression = fusedExp
  fmod.virtualOutputs = virtualOutputs

  -- remove n2 from graph
  -- here we assume that n2 only has n1 as parent, and n1 only has n2 as child
  -- need to improve this later...
  --n2.parents = {} 
  --n1.children = {}
  -- add n2's children as n1's
  --for i, n2child in n2.children do
  --  
  --end
  return true
end

--function fusion.fuseApply(nodes3)
--  -- fuse them...
--  -- for now, let's just look for a parent-child who are both applies, and fuse those
--  local didfuse = true
--  local fuseid = 1
--  while didfuse do
--    didfuse = false
--    ng3bg = nodes3:graph() -- not an nngraph, just a normal graph
--    ng3fg = ng3bg:reverse()
--    local n1 = nil
--    local n2 = nil -- this is the second, that eats the first
--    local n1_pos = nil
--  --  local n2_pos = nil
--    for i, node in ipairs(ng3fg.nodes) do
--      if n1 == nil then  -- since no 'break'
--        print('i', i, node.data.module)
--        if isApply(node.data.module) then
--          for j, child in ipairs(node.children) do  -- I know this is rubbish n-squared, fix this later..
--            if i1 == nil then
--              if isApply(child.data.module) then
--                n1 = node
--                n2 = child
--                n1_pos = i
--              end
--            end
--          end
--        end
--      end
--    end

--    if n1 ~= nil then
--      print('fusing... ==============================')
--      local n1_forward = n1.data.module.forwardExpression
--      local n2_forward = n2.data.module.forwardExpression
--      print('n1 forward', n1_forward)
--      print('n2 forward', n2_forward)

--      tempvar = 'sOutput' .. fuseid
--      fuseid = fuseid + 1
--      n1_forward = n1_forward:gsub('{{output}}', 'float ' .. tempvar)
--      n2_forward = n2_forward:gsub('{{input}}', tempvar)

--      print('n1 forward', n1_forward)
--      print('n2 forward', n2_forward)

--      local fused_forward_exp = n1_forward .. '\n' .. n2_forward
--      print('fused_forward', fused_forward_exp)

--      -- calculate forward first, to get the output values
--      -- do this by adding hte fused_forward_exp at the start
--      tempvar = 'sGradInput' .. fuseid
--      local n1_backward = n1.data.module.backwardExpression
--      local n2_backward = n2.data.module.backwardExpression
--      print('n1 backward', n1_backward)
--      print('n2 backward', n2_backward)
--      n2_backward = n2_backward:gsub('{{gradInput}}', 'float ' .. tempvar)
--      n1_backward = n1_backward:gsub('{{gradOutput}}', tempvar)

--      local fused_backwardonly_exp = n2_backward .. '\n' .. n1_backward
--      print('fused_backward', fused_backwardonly_exp)

--      local forward_then_backward = fused_forward_exp .. '\n' .. fused_backwardonly_exp

--      local fusedModule = nn.Apply(1, 1, fused_forward_exp, forward_then_backward)
--      fusedModule.backwardOnlyExpression = fused_backwardonly_exp
--      nodes3 = removeNodeByWalk(nodes3, n2.data)
--      n1.data.module = fusedModule
--      fusedModule.inputsNeeded = fusedModule.inputsNeeded or {}
--      fusedModule.gradOutputsNeeded = fusedModule.gradOutputsNeeded or {}
--      fusedModule.gradOutputsNeeded[n2.id] = true
--      fusedModule.gradOutputsNeeded[n1.id] = nil
--  --    table.insert(fusedModule.inputsNeeded, {id=)
--      didfuse = true
--    end
--  end
--  return nodes3
--end

return fusion

