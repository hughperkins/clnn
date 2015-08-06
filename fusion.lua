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

function fusion._createApply(params)
  local fusible = params.fusible
  local moduletype = params.moduletype
  local forwardExpression = params.forwardExpression
  local backwardExpression = params.backwardExpression
  local numInputs = params.numInputs or 1
  local numOutputs = params.numOutputs or 1

  if fusible.name == nil or fusible.name == '' then
    fusible.name = moduletype
  end
  fusible.feobj = {}
  fusible.beobj = {}
  local apply = nn.Apply(numInputs, numOutputs, '', '', moduletype)
  local transforms = {}
  for i=1, numInputs do
    transforms['input' .. i] = {src='input', inputIdx=i}
  end
  for i=1, numOutputs do
    transforms['output' .. i] = {src='output', outputIdx=i, virtualIdx=i}
  end
  table.insert(fusible.feobj, {template=forwardExpression, backward=backwardExpression,
    transforms=transforms})
  fusible.module = apply
  fusible.numInputs = numInputs
  fusible.numOutputs = numOutputs
  fusible.numVirtualOutputs = numOutputs
end

function fusion.convertToApply(fusible)
  local moduletype = torch.type(fusible.module)
  if moduletype == 'nn.Tanh' then
    fusion._createApply({fusible=fusible, moduletype=moduletype, 
      forwardExpression='{{output1}} = tanh({{input1}});', 
      backwardExpression='{{gradInput1}} = {{gradOutput1}} * (1 - {{output1}} * {{output1}});',
      numInputs=1, numOutputs=1})
  elseif moduletype == 'nn.Sigmoid' then
    fusion._createApply({fusible=fusible, moduletype=moduletype, 
      forwardExpression='{{output1}} = 1.f / (1.f + exp( - {{input1}}));', 
      backwardExpression='{{gradInput1}} = {{gradOutput1}} * {{output1}} * (1.f - {{output1}});',
      numInputs=1, numOutputs=1})
  elseif moduletype == 'nn.Exp' then
    fusion._createApply({fusible=fusible, moduletype=moduletype, 
      forwardExpression='{{output1}} = exp({{input1}});', 
      backwardExpression='{{gradInput1}} = {{gradOutput1}} * {{output1}};',
      numInputs=1, numOutputs=1})
  elseif moduletype == 'nn.Abs' then
    fusion._createApply({fusible=fusible, moduletype=moduletype, 
      forwardExpression='{{output1}} = fabs({{input1}});', 
      backwardExpression='{{gradInput1}} = {{input1}} < 0 ? - {{gradOutput1}} : {{gradOutput1}};',
      numInputs=1, numOutputs=1})
  elseif moduletype == 'nn.CAddTable' then
    fusion._createApply({fusible=fusible, moduletype=moduletype, 
      forwardExpression='{{output1}} = {{input1}} + {{input2}};', 
      backwardExpression='{{gradInput1}} = {{gradOutput1}}; {{gradInput2}} = {{gradOutput1}};',
      numInputs=2, numOutputs=1})
  elseif moduletype == 'nn.CMulTable' then
    fusion._createApply({fusible=fusible, moduletype=moduletype, 
      forwardExpression='{{output1}} = {{input1}} * {{input2}};', 
      backwardExpression='{{gradInput1}} = {{gradOutput1}}; {{gradInput2}} = {{gradOutput1}};',
      numInputs=2, numOutputs=1})
  elseif false and moduletype == 'nil' then
  end
end

function fusion.walkConvertToApply(fusibles)
  nn.Fusible.walkApply(fusibles, function(fusible)
    fusion.convertToApply(fusible)
  end)
end

-- assumes nothing has been fused yet, no existing
-- virtual idxes, everything is either an input or output,
-- for now
-- if there were virtual idxes already, I suppose we could
-- first walk over them, and start virtualidx numbering higher
-- than them
function fusion.walkAssignVirtualIdx(fusibles)
  local virtualIdx = 1
  nn.Fusible.walkApply(fusibles, function(fusible)
    if not fusion.isNodeApply(fusible) then
      return
    end
    print('fusible', fusible)
    local transformByOutputIdx = {}
    for k, transform in pairs(fusible.feobj[1].transforms) do
      if transform.outputIdx ~= nil then
        transformByOutputIdx[transform.outputIdx] = transform
        print('transform.outputIdx', transform.outputIdx)
      end
    end
    for i, output in ipairs(fusible.outputs) do
      print('output.outputIdx', output.outputIdx)
      local transform = transformByOutputIdx[output.outputIdx]
      transform.virtualIdx = virtualIdx
      if fusion.isNodeApply(output.child) then
        local cfo = nil
        for ck, ctrans in pairs(output.child.feobj[1].transforms) do
          if ctrans.inputIdx == output.inputIdx then
            ctrans.virtualIdx = virtualIdx
          end
        end
      end
      virtualIdx = virtualIdx + 1
    end
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
  print('incoming fe: ' .. fe)
  for target, value in pairs(feo.transforms) do
    if templateName == 'template' then
      if passName == 'forward' then
        -- === updateOutput forward section ====================
        if value.src == 'input' then
          if value.inputIdx ~= nil then
            fe = fe:gsub('{{' .. target .. '}}', value.src .. value.inputIdx .. '_data[n]')
          else
            fe = fe:gsub('{{' .. target .. '}}', 'virtualOutput' .. value.virtualIdx)
          end
--        elseif value.src == 'virtualInput' then
----          if target:find('output') ~= nil then
----            fe = fe:gsub('{{' .. target .. '}}', 'float ' .. value.src .. value.idx)
----          else
--            fe = fe:gsub('{{' .. target .. '}}', value.src .. value.virtualIdx)
----          end
        else
--        elseif value.src == 'output' then
          -- create virtualoutput, in case other operations need it, and also write
          -- to output
--          local virtualOutputIdx = dat.numVirtualOutputs + value.idx
          fe = fe:gsub('{{' .. target .. '}}', 'float virtualOutput' .. value.virtualIdx)
          if value.outputIdx ~= nil then
            local fe2 = value.src .. value.outputIdx .. '_data[n] = virtualOutput' .. value.virtualIdx .. ';'
            fe = fe .. '\n' .. fe2
          end
--        else
--          error('Unknown src ' .. value.src)
        end
      elseif false and passName == 'backward' then
        -- === updateGradInput, forward section ====================
        if value.src == 'input' then
          fe = fe:gsub('{{' .. target .. '}}', value.src .. value.inputIdx .. '_data[n]')
        elseif value.src == 'virtualInput' then
          fe = fe:gsub('{{' .. target .. '}}', 'float virtualOutput' .. value.virtualIdx)
        elseif value.src == 'virtualOutput' then
          if target:find('output') ~= nil then
            fe = fe:gsub('{{' .. target .. '}}', 'float ' .. value.src .. value.virtualIdx)
          else
            fe = fe:gsub('{{' .. target .. '}}', value.src .. value.idx)
          end
        elseif value.src == 'output' then
          -- convert to virtualOutput
--          local virtualOutputIdx = dat.numVirtualOutputs + value.idx
          fe = fe:gsub('{{' .. target .. '}}', 'float virtualOutput' .. value.virtualIdx)
        else
          error('Unknown src ' .. value.src)
        end
  --      if target:find('input') ~= nil then
  --        fe = fe:gsub('{{' .. target:gsub('input', 'gradInput') .. '}}', declaration .. value.src:gsub('input', 'gradInput') .. value.idx)
  --      elseif target:find('output') ~= nil then
  --        fe = fe:gsub('{{' .. target:gsub('output', 'gradOutput') .. '}}', declaration .. value.src:gsub('output', 'gradOutput') .. value.idx)
  --      end
      end
    elseif false and templateName == 'backward' then
      -- === updateGradInput, backward section ====================
--      print('  target=' .. target .. ' value.src=' .. value.src .. ' value.idx=' .. value.idx)
      if value.src == 'input' then
        fe = fe:gsub('{{' .. target:gsub('input', 'gradInput') .. '}}', 'gradInput' .. value.inputIdx .. '_data[n]')
      elseif value.src == 'virtualInput' then
        fe = fe:gsub('{{' .. target .. '}}', 'virtualOutput' .. value.virtualIdx)
      elseif value.src == 'output' then
--        local virtualOutputIdx = dat.numVirtualOutputs + value.idx
        fe = fe:gsub('{{' .. target .. '}}', 'virtualOutput' .. value.virtualIdx)
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
      print('    ->' .. fe)
    else
      -- error('Unknown template name %s', templateName)
    end
  end
  print('  fe', fe)
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
      be = ''
      mod:updateExpressions(mod.numInputs, mod.numOutputs, fe, be)
      print(mod.forwardKernel:getRenderedKernel())
--      print(mod.backwardKernel:getRenderedKernel())
    end
  end)
end

function fusion.doFuse(x)
  fusion.walkAssignVirtualIdx(x)
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

  parentIsWhichInput = nn.Fusible.getLinkPos(c.inputs, p)

  local pfo = pdat.feobj
  local cfo = cdat.feobj

  local newVirtualsBase = pdat.numVirtualOutputs + cdat.numVirtualOutputs
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

  -- renumber virtualIdxs of child
--  for i=1,#cfo do
--    local thiscfo = cfo[i]
--    for _, transform in pairs(thiscfo.transforms) do
--      if transform.virtualIdx ~= nil then
--        transform.virtualIdx = transform.virtualIdx + pdat.numVirtualOutputs
--      end
--    end
--  end
--   outputs from parent that go to child become virtual
  -- input to child too
  -- should have same virtual number
--  local numVirtualizedLinks = 0  -- this will increment by one each time we hit one link
  

--  -- links from parent to child become virtuals
  local ptransByOutputIdx = {}
  for i, pfo in ipairs(p.feobj) do
    for k, ptrans in pairs(pfo.transforms) do
      if ptrans.outputIdx ~= nil then
        ptransByOutputIdx[ptrans.outputIdx] = ptrans
      end
    end
  end
  local ctransByInputIdx = {}
  for i, cfo in ipairs(c.feobj) do
    for k, ctrans in pairs(cfo.transforms) do
      if ctrans.inputIdx ~= nil then
        ctransByInputIdx[ctrans.inputIdx] = ctrans
      end
    end
  end
  for i, output in ipairs(p.outputs) do
    if output.child == c then
      local ptrans = ptransByOutputIdx[output.outputIdx]
      ptrans.outputIdx = nil
      local ctrans = ctransByInputIdx[output.inputIdx]
      ctrans.inputIdx = nil
    end
  end

--  for i=1,#pfo do
--    local thispfo = pfo[i]
--    for _, transform in pairs(thispfo.transforms) do
--      if transform.src == 'output' and transform.outputIdx == childIndexInParent then
--        transform.outputIdx = nil
--      end
--    end
--    print('this pfo', thispfo)
--  end
--  local bumpParentInputsAmount = 0  -- increment this for each child input that is left of parent link
--  -- renumber inputs for child and parent, to preserve original relative order and not clobber each other
--  -- child input from parent becomes virtualoutput
--  for i=1,#cfo do
--    local thiscfo = cfo[i]
--    for _, transform in pairs(thiscfo.transforms) do
--      if transform.src == 'input' and transform.inputIdx == parentIndexInChild then
--        transform.inputIdx = nil
----        transform.src = 'virtualInput'
--        transform.virtualIdx = newVirtualsBase + 1
----        transform.idx = virtualOutputBase + 1
--      end
--      if transform.src == 'input' and transform.inputIdx ~= nil and transform.inputIdx ~= parentIndexInChild then
--        if transform.inputIdx > parentIndexInChild then
----          transform.idx = transform.idx + pmod.numInputs - 1
--        else
--          bumpParentInputsAmount = bumpParentInputsAmount + 1
--        end
--      end
--    end
--  end
--  for i=1,#pfo do
--    local thispfo = pfo[i]
--    for _, transform in pairs(thispfo.transforms) do
--      if transform.src == 'input' and transform.inputIdx ~= nil then
--        transform.inputIdx = transform.inputIdx + bumpParentInputsAmount
--      end
--    end
--  end
--  -- move outputs from child to parent, merging any duplicates
--  local parentOuts = {} -- set of parent output fusibles, for quick lookup
--  for j, parentOut in ipairs(pdat.outputs) do
--    parentOuts[parentOut.child] = j
--  end
--  for i, childOut in ipairs(cdat.outputs) do
--    if parentOuts[childOut.child] ~= nil then
--      -- merge them
--    else
--      -- move from child to parent
--      
--    end
--  end

  local fusedfos = {}
  for i=1,#pfo do
    local thispfo = pfo[i]
    table.insert(fusedfos, thispfo)
  end
  for i=1,#cfo do
    local thiscfo = cfo[i]
    table.insert(fusedfos, thiscfo)
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

