fusion = {}

-- this assumes inverted, wrt g.bg, ie root of nodes, highest parent, is x, input
-- and bottom, the child, is results
-- in other words, its inverted wrt the graph created by doing like nn.Tanh()(nn.Sigmoid()()),
-- which would make tanh a parent of sigmoid.  here we expect tanh to be the child of sigmoid,
-- in this example

local ngh = require('nodeGraphHelper')

function fusion.isNodeApply(node)
  if node.data.module == nil then
    return false
   end
  if torch.type(node.data.module) == 'nn.Apply' then
    return true
  end
  return false
end

function fusion.isModuleApply(module)
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
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = tanh({{input1}});', transforms={input1={src='input',idx=1}, output1={src='output',idx=1}}})
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
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = 1.f / (1.f + exp( - {{input1}}));', transforms={input1={src='input', idx=1}, output1={src='output', idx=1}}})
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
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = exp({{input1}});', transforms={input1={src='input', idx=1}, output1={src='output', idx=1}}})
    table.insert(dat.beobj, {template='{{gradInput1}} = {{gradOutput1}} * {{output1}};',
      transforms={gradInput1='gradInput1', gradOutput1='gradOutput1', output1='output1'}})
    local apply = nn.Apply(1, 1, [[
      {{output}} =  exp({{input}});
    ]], [[
      {{gradInput}} = {{gradOutput}} * {{output}};
    ]], moduletype)
    node.data.module = apply
  elseif moduletype == 'nn.Abs' then
    local dat = node.data
    dat.name = moduletype
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = fabs({{input1}});', transforms={input={src='input', idx=1}, output1={src='output', idx=1}}})
    table.insert(dat.beobj, {template='{{gradInput1}} = {{input1}} < 0 ? - {{gradOutput1}} : {{gradOutput1}};',
      transforms={gradInput1='gradInput1', gradOutput1='gradOutput1', input1='input1'}})
    local apply = nn.Apply(1, 1, [[
      {{output}} =  fabs({{input}});
    ]], [[
      {{gradInput}} = {{input}} < 0 ? - {{gradOutput}} : {{gradOutput}};
    ]], moduletype)
    node.data.module = apply
  elseif moduletype == 'nn.CAddTable' then
    local dat = node.data
    dat.name = moduletype
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = {{input1}} + {{input2}};', transforms={input1={src='input', idx=1}, input2={src='input', idx=2}, output1={src='output', idx=1}}})
    table.insert(dat.beobj, {template=
[[{{gradInput1}} = {{gradOutput}};
{{gradInput2}} = {{gradOutput}};]],
      transforms={gradInput1='gradInput1', gradInput2='gradInput2', gradOutput1='gradOutput1'}})
    local apply = nn.Apply(2, 1, [[
      {{output}} = {{input1}} + {{input2}};
    ]], [[
      {{gradInput1}} = {{gradOutput}};
      {{gradInput2}} = {{gradOutput}};
    ]], moduletype)
    node.data.module = apply
  elseif moduletype == 'nn.CMulTable' then
    local dat = node.data
    dat.name = moduletype
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = {{input1}} * {{input2}};', transforms={input1={src='input', idx=1}, input2={src='input', idx=2}, output={src='output', idx=1}}})
    table.insert(dat.beobj, {template=
[[{{gradInput1}} = {{gradOutput1}};
{{gradInput2}} = {{gradOutput}};]],
      transforms={gradInput1='gradInput1', gradInput2='gradInput2', gradOutput='gradOutput1'}})
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
    if fusion.isNodeApply(node) then
      for j, child in ipairs(node.children) do  -- I know this is rubbish n-squared, fix this later..
        if fusion.isNodeApply(child) then
          n1 = node
          n2 = child
          return
        end
      end
    end
  end)
  return n1, n2
end

function fusion.expandTemplate(feo)
  fe = feo.template
  for target, value in pairs(feo.transforms) do
    fe = fe:gsub('{{' .. target .. '}}', value.src .. value.idx)
  end
  --print(fe)
  return fe
end

function fusion.generateKernels(x)
  ngh.walkApply(x, function(node)
--    print('apply node', node.data.module)
    print('node ' .. ngh.nodeGetName(node))
    if fusion.isNodeApply(node) then
      fe = ''
      for i, onefe in ipairs(node.data.feobj) do
        fe = fe .. fusion.expandTemplate(onefe) .. '\n'
      end
      print('fe', fe)
    end
  end)
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
--  print('virtualOutputs before =', virtualOutputs)
  -- virtualOutputs = virtualOutputs + mod1.numOutputs
  -- mod1.virtualOutputs = virtualOutputs

  -- observations:
  -- ALL parent's outputs go to child
  -- but normally parent will just have one unique output (could go to multiple children)
  -- child might have more than one input
  -- need to find which input comes from child
  parentIsWhichInput = ngh.getLinkPos(c.parents, p)
--  print('parent input:', parentIsWhichInput)

  local pfo = pdat.feobj
  local cfo = cdat.feobj
--  local fusedfo = {}
--  for i, feo in ipairs(pfo) do
----    print('inserting', feo)
--    table.insert(fusedfo, feo)
--  end
--  for i, feo in ipairs(cfo) do
----    print('inserting', feo)
--    table.insert(fusedfo, feo)
--  end

  -- for all child inputs which dont come from parent, and there will be exactly one from
  -- parent, add them to parent inputs
  local newNumInputs = pmod.numInputs + cmod.numInputs - 1  -- -1, because one came from parent
  -- need to bump all child inputs by the number of parent inputs
--  for i=1, c.numInputs do
--    for j=1, #cfo do
--      if cfo[j].transforms['input'] == 'input' .. i then
--        cfo[j].transforms['input'] = 'input' .. (i + p.numInputs)
--      end
--      k = 1
--      while cfo[j].transforms['input' .. k] ~= nil do
--        cfo[j].transforms['input' .. k] = 'in
--        k = k + 1
--      end
--      if cfo[j].transforms['input' .. i] ~= nil then
--        cfo[j].transforms['input' .. i]
--      end
--    end
--  end

  local virtualOutputBase = pdat.numVirtualOutputs + cdat.numVirtualOutputs
  local newNumVirtualOutputs = pdat.numVirtualOutputs + cdat.numVirtualOutputs + pmod.numOutputs
--  pfo[#pfo].transforms.output = 'float virtualOutput' .. virtualOutputs

  -- actions on merge:
  -- - virtualoutputs of child will need to be renumbered, so dont clobber parent (ie translated by
  --   number of idx equal to number of parent virtualoutputs)
  -- - there is one child output that feeds into child.  this will create one additional virtuaoutput
  --   - we should find what is the input index for child, and output index for parent
  local childIndexInParent = ngh.getLinkPos(p.children, c)
  local parentIndexInChild = ngh.getLinkPos(c.parents, p)
  print('link pos childinparent=' .. childIndexInParent .. ' parentinchild=' .. parentIndexInChild)
  local fusedfos = {}
  for i=1,#pfo do
    local thispfo = pfo[i]
    for _, transform in pairs(thispfo.transforms) do
      if transform.src == 'output' and transform.idx == childIndexInParent then
        transform.src = 'virtualOutput'
        transform.idx = transform.idx + virtualOutputBase
      end
    end
    print('this pfo', thispfo)
    table.insert(fusedfos, thispfo)
  end
  for i=1,#cfo do
    local thiscfo = cfo[i]
    for _, transform in pairs(thiscfo.transforms) do
      if transform.src == 'input' and transform.idx == parentIndexInChild then
        transform.src = 'virtualOutput'
        transform.idx = transform.idx + virtualOutputBase
      end
    end
    table.insert(fusedfos, thiscfo)
  end

--  if cfo[1].transforms.input.src == 'input' then
--    cfo[1].transforms.src = 'virtualOutput'
--    cfo[1].transforms.idx = virtualOutputs
--  end
--  if cfo[1].transforms['input' .. parentIsWhichInput] ~= nil then
--    cfo[1].transforms['input' .. parentIsWhichInput] = 'virtualOutput' .. virtualOutputs
--  end
  for o=1,pmod.numOutputs do
--    cfo
--    cf = cf:gsub('{{output' .. o .. '}}', 'float {{virtualOut' .. virtualOutputs .. '}}')
--    pf = pf:gsub('{{input' .. o .. '}}', '{{virtualOut' .. virtualOutputs .. '}}')
  end

  local fused = ngh.reduceEdge(p, c)
  local fdat = fused.data
  fdat.feobj = fusedfos
--  fdat.numInputs = newNumInputs
  fdat.numOutputs = 1  -- I guess???  might generalize this a bit, later...
  local fmod = fdat.module
  fmod.numInputs = pmod.numInputs
  fmod.numOutputs = pmod.numOutputs + cmod.numOutputs - 1
  fmod.forwardExpression = fusedExp
  fdat.numVirtualOutputs = newNumVirtualOutputs
  ngh.nodeSetName(fused, ngh.nodeGetName(c) .. '.' .. ngh.nodeGetName(p))

  return true
end

return fusion

