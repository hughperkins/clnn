fusion = {}

-- this assumes inverted, wrt g.bg, ie root of fusibles, highest parent, is x, input
-- and bottom, the child, is results
-- in other words, its inverted wrt the graph created by doing like nn.Tanh()(nn.Sigmoid()()),
-- which would make tanh a parent of sigmoid.  here we expect tanh to be the child of sigmoid,
-- in this example

--local ngh = require('fusibleGraphHelper')

function fusion.isNodeApply(fusible)
  if fusible.module == nil then
    return false
   end
  if torch.type(fusible.module) == 'nn.Apply' then
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

--function fusion.initClonedOutputs(fusible)
--  local dat = fusible
--  dat.outputs = {}
--  local outputs = dat.outputs
--  for i, child in ipairs(fusible.children) do
--    -- inputIdx is idx of the input into child fusible
--    -- to get this, we assume that all inputs into child are 
--    -- unique, and we look at the sequence number in the 
--    -- child.parents table
--    -- we are only going to store an outputs table, no inputs table
--    -- we can get the outputs table from the child, via the parent link
--    -- note that child can have multiple inputs from each parent
--    local inputIdx = nn.Fusible.getLinkPos(child.parents, fusible)
--    local output = {outputIdx=1, child=child, inputIdx=inputIdx}
--    table.insert(outputs, output)
--  end
--end

function fusion.convertToApply(fusible)
  local moduletype = torch.type(fusible.module)
  if moduletype == 'nn.Tanh' then
    local dat = fusible
    dat.name = moduletype
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = tanh({{input1}});',
      transforms={input1={src='input',idx=1}, output1={src='output',idx=1}},
      backward='{{gradInput1}} = {{gradOutput1}} * (1 - {{output1}} * {{output1}});'})
    table.insert(dat.beobj, {template='{{gradInput1}} = {{gradOutput1}} * (1 - {{output1}} * {{output1}});',
      transforms={gradInput1='gradInput', gradOutput1='gradOutput', output1='output'}})
    local apply = nn.Apply(1, 1, [[
      {{output}} = tanh({{input}});
    ]], [[
      {{gradInput}} = {{gradOutput}} * (1 - {{output}} * {{output}});
    ]], moduletype)
    fusible.module = apply
--    fusion.initClonedOutputs(fusible)
  elseif moduletype == 'nn.Sigmoid' then
    local dat = fusible
    dat.name = moduletype
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = 1.f / (1.f + exp( - {{input1}}));',
      transforms={input1={src='input', idx=1}, output1={src='output', idx=1}},
      backward='{{gradInput1}} = {{gradOutput1}} * {{output1}} * (1.f - {{output1}});'})
    table.insert(dat.beobj, {template='{{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});',
      transforms={gradInput='gradInput', gradOutput='gradOutput', output='output'}})
    local apply = nn.Apply(1, 1, [[
      {{output}} =  1.f / (1.f + exp( - {{input}}));
    ]], [[
      {{gradInput}} = {{gradOutput}} * {{output}} * (1.f - {{output}});
    ]], moduletype)
    fusible.module = apply
--    fusion.initClonedOutputs(fusible)
  elseif moduletype == 'nn.Exp' then
    local dat = fusible
    dat.name = moduletype
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = exp({{input1}});',
      transforms={input1={src='input', idx=1}, output1={src='output', idx=1}},
      backward='{{gradInput1}} = {{gradOutput1}} * {{output1}};'})
    table.insert(dat.beobj, {template='{{gradInput1}} = {{gradOutput1}} * {{output1}};',
      transforms={gradInput1='gradInput1', gradOutput1='gradOutput1', output1='output1'}})
    local apply = nn.Apply(1, 1, [[
      {{output}} =  exp({{input}});
    ]], [[
      {{gradInput}} = {{gradOutput}} * {{output}};
    ]], moduletype)
    fusible.module = apply
--    fusion.initClonedOutputs(fusible)
  elseif moduletype == 'nn.Abs' then
    local dat = fusible
    dat.name = moduletype
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = fabs({{input1}});',
      transforms={input={src='input', idx=1}, output1={src='output', idx=1}},
      backward='{{gradInput1}} = {{input1}} < 0 ? - {{gradOutput1}} : {{gradOutput1}};'})
    table.insert(dat.beobj, {template='{{gradInput1}} = {{input1}} < 0 ? - {{gradOutput1}} : {{gradOutput1}};',
      transforms={gradInput1='gradInput1', gradOutput1='gradOutput1', input1='input1'}})
    local apply = nn.Apply(1, 1, [[
      {{output}} =  fabs({{input}});
    ]], [[
      {{gradInput}} = {{input}} < 0 ? - {{gradOutput}} : {{gradOutput}};
    ]], moduletype)
    fusible.module = apply
--    fusion.initClonedOutputs(fusible)
  elseif moduletype == 'nn.CAddTable' then
    local dat = fusible
    dat.name = moduletype
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = {{input1}} + {{input2}};',
      transforms={input1={src='input', idx=1}, input2={src='input', idx=2}, output1={src='output', idx=1}},
      backward='{{gradInput1}} = {{gradOutput1}}; {{gradInput2}} = {{gradOutput1}};'})
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
    fusible.module = apply
--    fusion.initClonedOutputs(fusible)
  elseif moduletype == 'nn.CMulTable' then
    local dat = fusible
    dat.name = moduletype
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = {{input1}} * {{input2}};',
      transforms={input1={src='input', idx=1}, input2={src='input', idx=2}, output1={src='output', idx=1}},
      backward='{{gradInput1}} = {{gradOutput1}}; {{gradInput2}} = {{gradOutput1}};'})
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
    fusible.module = apply
--    fusion.initClonedOutputs(fusible)
  elseif false and moduletype == 'nil' then
    local dat = fusible
    dat.name = moduletype
    dat.numVirtualOutputs = 0
    dat.feobj = {}
    dat.beobj = {}
    table.insert(dat.feobj, {template='{{output1}} = {{input1}};',
      transforms={input1={src='input', idx=1}, output1={src='output', idx=1}},
      backward='{{gradInput1}} = {{gradOutput1}};'})
    table.insert(dat.beobj, {template=
[[{{gradInput1}} = {{gradOutput1}};
{{gradInput2}} = {{gradOutput}};]],
      transforms={gradInput1='gradInput1', gradInput2='gradInput2', gradOutput='gradOutput1'}})
    local apply = nn.Apply(1, 1, [[
      {{output}} = {{input1}} + {{input2}};
    ]], [[
      {{gradInput1}} = {{gradOutput}};
      {{gradInput2}} = {{gradOutput}};
    ]], moduletype)
    fusible.module = apply
--    fusion.initClonedOutputs(fusible)
  end
end

function fusion.walkConvertToApply(fusibles)
  nn.Fusible.walkApply(fusibles, function(fusible)
    fusion.convertToApply(fusible)
  end)
end

function fusion.reverseWalkConvertToApply(x)
  nn.Fusible.reverseWalkApply(x, function(fusible)
    fusion.convertToApply(fusible)
  end)
end

function fusion.getFusiblePair(x)
  local n1 = nil
  local n2 = nil
  nn.Fusible.walkApply(x, function(fusible)
    if n1 ~= nil then
      return
    end
    if fusion.isNodeApply(fusible) then
      for j, output in ipairs(fusible.outputs) do  -- I know this is rubbish n-squared, fix this later..
        if fusion.isNodeApply(output.child) then
          n1 = fusible
          n2 = output.child
          return
        end
      end
    end
  end)
  return n1, n2
end

function fusion.expandTemplate(dat, feo, templateName, passName)
  local fe = feo[templateName]
--  print('incoming fe: ' .. fe)
  for target, value in pairs(feo.transforms) do
    if templateName == 'template' then
      if passName == 'forward' then
        -- === updateOutput forward section ====================
        if value.src == 'input' then
          fe = fe:gsub('{{' .. target .. '}}', value.src .. value.idx .. '_data[n]')
        elseif value.src == 'virtualOutput' then
          if target:find('output') ~= nil then
            fe = fe:gsub('{{' .. target .. '}}', 'float ' .. value.src .. value.idx)
          else
            fe = fe:gsub('{{' .. target .. '}}', value.src .. value.idx)
          end
        else
          fe = fe:gsub('{{' .. target .. '}}', value.src .. value.idx .. '_data[n]')
        end
      elseif passName == 'backward' then
        -- === updateGradInput, forward section ====================
        if value.src == 'input' then
          fe = fe:gsub('{{' .. target .. '}}', value.src .. value.idx .. '_data[n]')
        elseif value.src == 'virtualOutput' then
          if target:find('output') ~= nil then
            fe = fe:gsub('{{' .. target .. '}}', 'float ' .. value.src .. value.idx)
          else
            fe = fe:gsub('{{' .. target .. '}}', value.src .. value.idx)
          end
        elseif value.src == 'output' then
          -- convert to virtualOutput
          local virtualOutputIdx = dat.numVirtualOutputs + value.idx
          fe = fe:gsub('{{' .. target .. '}}', 'float virtualOutput' .. virtualOutputIdx)
        end
  --      if target:find('input') ~= nil then
  --        fe = fe:gsub('{{' .. target:gsub('input', 'gradInput') .. '}}', declaration .. value.src:gsub('input', 'gradInput') .. value.idx)
  --      elseif target:find('output') ~= nil then
  --        fe = fe:gsub('{{' .. target:gsub('output', 'gradOutput') .. '}}', declaration .. value.src:gsub('output', 'gradOutput') .. value.idx)
  --      end
      end
    elseif templateName == 'backward' then
      -- === updateGradInput, backward section ====================
--      print('  target=' .. target .. ' value.src=' .. value.src .. ' value.idx=' .. value.idx)
      if value.src == 'input' then
        fe = fe:gsub('{{' .. target:gsub('input', 'gradInput') .. '}}', 'gradInput' .. value.idx .. '_data[n]')
      elseif value.src == 'output' then
        local virtualOutputIdx = dat.numVirtualOutputs + value.idx
        fe = fe:gsub('{{' .. target .. '}}', 'virtualOutput' .. virtualOutputIdx)
--        fe = fe:gsub(target:gsub('output', 'gradOutput'), 'gradOutput' .. value.idx)
      elseif value.src == 'virtualOutput' then
        if target:find('input') ~= nil then
          fe = fe:gsub('{{' .. target:gsub('input', 'gradInput') .. '}}', 'float ' .. value.src:gsub('virtualOutput', 'virtualGradInput') .. value.idx)
        else
          fe = fe:gsub('{{' .. target:gsub('output', 'gradOutput') .. '}}', value.src:gsub('virtualOutput', 'virtualGradInput') .. value.idx)
          fe = fe:gsub('{{' .. target .. '}}', value.src .. value.idx)
        end

--        fe = fe:gsub('{{' .. target .. '}}', value.src .. value.idx .. '_data[n]')
--      elseif value.src == 'virtualOutput' then
--        if target:find('output') ~= nil then
--          fe = fe:gsub('{{' .. target .. '}}', 'float ' .. value.src .. value.idx)
--        else
--          fe = fe:gsub('{{' .. target .. '}}', value.src .. value.idx)
--        end
--      else
--        fe = fe:gsub('{{' .. target .. '}}', value.src .. value.idx .. '_data[n]')
      else
        error('unknown value.src %s', value.src)
      end
--      print('    ->' .. fe)
    else
      error('Unknown template name %s', templateName)
    end
  end
  --print(fe)
  return fe
end

function fusion.generateKernels(x)
  local seen = {}
  nn.Fusible.walkApply(x, function(fusible)
    if seen[fusible] then
      return
    end
    seen[fusible] = true
--    print('apply fusible', fusible.module)
--    print('fusible ' .. nn.Fusible.fusibleGetName(fusible))
    if fusion.isNodeApply(fusible) then
      local fe = ''
      local be = ''
      for i, onefe in ipairs(fusible.feobj) do
        fe = fe .. fusion.expandTemplate(fusible, onefe, 'template', 'forward') .. '\n'
        be = be .. fusion.expandTemplate(fusible, onefe, 'template', 'backward') .. '\n'
      end
      for i=#fusible.feobj, 1, -1 do
        local onefe = fusible.feobj[i]
--        print('onefe', onefe)
--      for i, onefe in ipairs(fusible.feobj) do
        be = be .. fusion.expandTemplate(fusible, onefe, 'backward', 'backward') .. '\n'
      end
--      print('fe', fe)
--      print('be', be)
      local dat = fusible
      local mod = dat.module
      mod:updateExpressions(mod.numInputs, mod.numOutputs, fe, be)
--      print(mod.forwardKernel:getRenderedKernel())
--      print(mod.backwardKernel:getRenderedKernel())
    end
  end)
end

function fusion.doFuse(x)
  while fusion.doFuseIteration(x) do
  end
end

function getChildIndexInParent(parent, child)
  for i, output in ipairs(parent.outputs) do
    if output.child == child then
      return i
    end
  end
end

-- since we inverted this:
-- child function is applied to result of parent function
-- we're going to move/fuse/merge all chid things into parent
-- then throw away the child
function fusion.doFuseIteration(x)
  p, c = fusion.getFusiblePair(x)
  if p == nil then
    return false
  end

  local pdat = p
  local cdat = c
  local pmod = pdat.module
  local cmod = cdat.module

  local p_inputs = pmod.numInputs
  local c_inputs = cmod.numInputs
  local p_outputs = pmod.numOutputs
  local c_outputs = cmod.numOutputs

  parentIsWhichInput = nn.Fusible.getLinkPos(c.inputs, p)

  local pfo = pdat.feobj
  local cfo = cdat.feobj

  -- for all child inputs which dont come from parent, and there will be exactly one from
  -- parent, add them to parent inputs
  local newNumInputs = pmod.numInputs + cmod.numInputs - 1  -- -1, because one came from parent
  local newNumOutputs = pmod.numOutputs + cmod.numOutputs - 1  -- -1, because one came from parent

  local virtualOutputBase = pdat.numVirtualOutputs + cdat.numVirtualOutputs
  local newNumVirtualOutputs = pdat.numVirtualOutputs + cdat.numVirtualOutputs + pmod.numOutputs

  -- actions on merge:
  -- - virtualoutputs of child will need to be renumbered, so dont clobber parent (ie translated by
  --   number of idx equal to number of parent virtualoutputs)
  -- - there is one parent output that feeds into child.  this will create one additional virtuaoutput
  --   - we should find what is the input index for child, and output index for parent
  -- - input idxes in child need to be shifted by amount equal to number of inputs in parent - 1
  local childIndexInParent = getChildIndexInParent(p, c)
  local parentIndexInChild = nn.Fusible.getLinkPos(c.inputs, p)
  print('link pos childinparent=' .. childIndexInParent .. ' parentinchild=' .. parentIndexInChild)
  local fusedfos = {}

  -- renumber virtualOutputs of child
  for i=1,#cfo do
    local thiscfo = cfo[i]
    for _, transform in pairs(thiscfo.transforms) do
      if transform.src == 'virtualOutput' then
        transform.idx = transform.idx + pdat.numVirtualOutputs
      end
    end
  end
  -- output from parent to child becomes virtualoutput
  for i=1,#pfo do
    local thispfo = pfo[i]
    for _, transform in pairs(thispfo.transforms) do
      if transform.src == 'output' and transform.idx == childIndexInParent then
        transform.src = 'virtualOutput'
        transform.idx = virtualOutputBase + 1
      end
    end
    print('this pfo', thispfo)
    table.insert(fusedfos, thispfo)
  end
  local bumpParentInputsAmount = 0  -- increment this for each child input that is left of parent link
  -- renumber inputs for child and parent, to preserve original relative order and not clobber each other
  -- child input from parent becomes virtualoutput
  for i=1,#cfo do
    local thiscfo = cfo[i]
    for _, transform in pairs(thiscfo.transforms) do
      if transform.src == 'input' and transform.idx == parentIndexInChild then
        transform.src = 'virtualOutput'
        transform.idx = virtualOutputBase + 1
      end
      if transform.src == 'input' and transform.idx ~= parentIndexInChild then
        if transform.idx > parentIndexInChild then
          transform.idx = transform.idx + pmod.numInputs - 1
        else
          bumpParentInputsAmount = bumpParentInputsAmount + 1
        end
      end
    end
    table.insert(fusedfos, thiscfo)
  end
  for i=1,#pfo do
    local thispfo = pfo[i]
    for _, transform in pairs(thispfo.transforms) do
      if transform.src == 'input' then
        transform.idx = transform.idx + bumpParentInputsAmount
      end
    end
  end
  -- move outputs from child to parent, merging any duplicates
  local parentOuts = {} -- set of parent output fusibles, for quick lookup
  for j, parentOut in ipairs(pdat.outputs) do
    parentOuts[parentOut.child] = j
  end
  for i, childOut in ipairs(cdat.outputs) do
    if parentOuts[childOut.child] ~= nil then
      -- merge them
    else
      -- move from child to parent
      
    end
  end

  local fused = nn.Fusible.reduceEdge(p, c)
  local fdat = fused
  fdat.feobj = fusedfos
  fdat.id = pdat.id .. '.' .. cdat.id
  local fmod = fdat.module
  fmod.numInputs = newNumInputs
  fmod.numOutputs = newNumOutputs
  fmod.forwardExpression = fusedExp
  fdat.numVirtualOutputs = newNumVirtualOutputs
  fused.name = c.name .. '.' .. p.name

  return true
end

return fusion

