function nn.Fusible.addLink(targetTable, value)
  if nn.Fusible.getLinkPos(targetTable, value) == nil then
    table.insert(targetTable, value)
  end
end

function nn.Fusible.removeLink(targetTable, value)
  local pos = nn.Fusible.getLinkPos(targetTable, value)
  if pos ~= nil then
    table.remove(targetTable, pos)
  end
end

function nn.Fusible.addEdge(parent, child)
--  nn.Fusible.addLink(parent.children, child)
  nn.Fusible.addLink(child.inputs, parent)
  table.insert(parent.output, {child=child, outputIdx=#parent.outputs + 1, inputIdx=#child.inputs})
end

function nn.Fusible.removeEdge(parent, child)
  nn.Fusible.removeLink(parent.children, child)
  nn.Fusible.removeLink(child.inputs, parent)
end

-- preserves the pos of the edge in the parent.children table
-- pos of the edge in the child tables will changes, since
-- edge didnt exist before
function nn.Fusible.moveEdgeChild(parent, oldChild, newChild)
  local posInParent = nn.Fusible.getLinkPos(parent.children, oldChild)
  parent.children[posInParent] = newChild
  nn.Fusible.removeLink(oldChild.inputs, parent)
  nn.Fusible.addLink(newChild.inputs, parent)
end

function nn.Fusible.moveEdgeParent(child, oldParent, newParent)
  local posInChild = nn.Fusible.getLinkPos(child.inputs, oldParent)
  child.inputs[posInChild] = newParent
  nn.Fusible.removeLink(oldParent.children, child)
  nn.Fusible.addLink(newParent.children, child)
end

function nn.Fusible.getChildOutputIndexInParent(parent, child)
  for i, output in ipairs(parent.outputs) do
    if output.child == child then
      return i
    end
  end
end

function nn.Fusible.getParentOutput(child, parent)
  for i, output in ipairs(parent.outputs) do
    if output.child == child then
      return output
    end
  end
end

function nn.Fusible.getOutputForChild(parent, child)
  for i, output in ipairs(parent.outputs) do
    if output.child == child then
      return output
    end
  end
end

function nn.Fusible.reduceEdge(parent, child)
--  nn.Fusible.removeEdge(parent, child)
  -- all children of the child should move to parent
  -- since the child output will be differnet from any previous outputs from the parent
  -- this will always be a brandnew output from the parent
  -- but we might want to renumber the existing output fro mthe parent, to conserve
  -- the same left-right sequence, when we look at a picture of hte graph
  -- ie outputs to the right of the child will be pushed one place ot hte right, for 
  -- each of the outputs from the child
  -- lets renumber the parent outputs first
  local shiftParent = 0
  local injectedChildOutputs = false
  local newParentOutputs = {}
  for i, parentOutput in ipairs(parent.outputs) do
    if parentOutput.child == child then
      shiftParent = shiftParent - 1
      if not injectedChildOutputs then
        for j, childOutput in ipairs(child.outputs) do
          -- assume no child outputs are the parent, since its a DAG
          -- so just add them all to the parent at this point
          table.insert(newParentOutputs, {child=childOutput.child, inputIdx=childOutput.inputIdx, outputIdx=i + j - 1})
          -- need to update the childchild's input entry to point to parent now
          childOutput.child.inputs[childOutput.inputIdx] = parent
          shiftParent = shiftParent + 1
        end
      end
    else
      parentOutput.outputIdx = parentOutput.outputIdx + shiftParent
    end
  end
  for i, newOutput in ipairs(newParentOutputs) do
    table.insert(parent.outputs, newOutput)
  end

--  local numParentOutputs = parent.numOutputs
--  for i, childOutput in ipairs(child.outputs) do
--    local childChild = childOutput.child
--    local parentOutput = childChild:getParentOutput([childOutput.inputIdx]
--    nn.Fusible.moveEdgeParent(childoutput.child, child, parent)
--  end  

  -- all inputs to child should move to parent, unless already present
  -- on parent
  -- the inputs on the parent should maintain relative order as follows:
  -- [inputs to child left of input from parent] [parent inputs] [inputs to child right of input from parent]
--  shiftChildLinks = 0
  local newParentInputs = {}
  local insertedParentInputs = false
  for i, input in ipairs(child.inputs) do
    if input == parent then
      -- insert the existing parent inputs here
      if not insertedParentInputs then
        for j, parentInput in ipairs(parent.inputs) do
          table.insert(newParentInputs, parentInput)
        end
      end
      insertedParentInputs = true
    else
      table.insert(newParentInputs, input)
      -- need to go and find the corresponding output, and redirect it to parent
      local childParentOutput = input:getOutputForChild(child)
      childParentOutput.child = parent
      childParentOutput.inputIdx = #newParentInputs
    end
  end
  parent.inputs = newParentInputs

  -- parent should no longer link to child
  -- there might be several links, we should remove all
  -- should walk in inverse order...
  for i=#parent.outputs, 1, -1 do
    local output = parent.outputs[i]
    if output.child == child then
      table.remove(parent.outputs, i)
    end
  end

-- need to keep order of parent links, so first first parent index from child
--  local parentIndexFromChild = nn.Fusible.getLinkPos(child.inputs, parent)
--  print('parentIndexFromChild', parentIndexFromChild)
--  -- all child parents less than this need to be inserted behind any parent links from parent
--  -- the rest go after the existing parent links from parent
--  for i, childparent in ipairs(child.inputs) do
--    if childparent ~= parent then
--      local childPosInChildParent = nn.Fusible.getChildIndexInParent(childparent, child)
--      if i < parentIndexFromChild then
--        table.insert(parent.inputs, i, childparent)
--      elseif i > parentIndexFromChild then
--        table.insert(parent.inputs, childparent)
--      end
--      local output = child:getParentOutput(childparent)
--      output.child = parent
--      -- need to update inputIdx almost certianly...
--    end
--  end
--  nn.Fusible.removeEdge(parent, child)

--  for i=#child.inputs,1,-1 do
--    local childparent = child.inputs[i]
--    nn.Fusible.addEdge(childparent, parent)
--  end
--  for i=#child.inputs,1,-1 do
--    local childparent = child.inputs[i]
--    nn.Fusible.removeEdge(childparent, child)
--  end
  return parent
end

