function fusibles.addLink(targetTable, value)
  if fusibles.getLinkPos(targetTable, value) == nil then
    table.insert(targetTable, value)
  end
end

function fusibles.removeLink(targetTable, value)
  local pos = fusibles.getLinkPos(targetTable, value)
  if pos ~= nil then
    table.remove(targetTable, pos)
  end
end

function fusibles.addEdge(parent, child)
  fusibles.addLink(parent.children, child)
  fusibles.addLink(child.parents, parent)
end

function fusibles.removeEdge(parent, child)
  fusibles.removeLink(parent.children, child)
  fusibles.removeLink(child.parents, parent)
end

-- preserves the pos of the edge in the parent.children table
-- pos of the edge in the child tables will changes, since
-- edge didnt exist before
function fusibles.moveEdgeChild(parent, oldChild, newChild)
  local posInParent = fusibles.getLinkPos(parent.children, oldChild)
  parent.children[posInParent] = newChild
  fusibles.removeLink(oldChild.parents, parent)
  fusibles.addLink(newChild.parents, parent)
end

function fusibles.moveEdgeParent(child, oldParent, newParent)
  local posInChild = fusibles.getLinkPos(child.parents, oldParent)
  child.parents[posInChild] = newParent
  fusibles.removeLink(oldParent.children, child)
  fusibles.addLink(newParent.children, child)
end

function fusibles.reduceEdge(parent, child)
--  fusibles.removeEdge(parent, child)
  -- all children of the child should move to parent
  for i, childchild in ipairs(child.children) do
    fusibles.moveEdgeParent(childchild, child, parent)
  end

  -- all parent links on the child should move to parent, unless already present
  -- on parent
  -- need to keep order of parent links, so first first parent index from child
  local parentIndexFromChild = fusibles.getLinkPos(child.parents, parent)
  print('parentIndexFromChild', parentIndexFromChild)
  -- all child parents less than this need to be inserted behind any parent links from parent
  -- the rest go after the existing parent links from parent
  for i, childparent in ipairs(child.parents) do
    if childparent ~= parent then
      local childPosInChildParent = fusibles.getLinkPos(childparent.children, child)
      if i < parentIndexFromChild then
        table.insert(parent.parents, i, childparent)
      elseif i > parentIndexFromChild then
        table.insert(parent.parents, childparent)
      end
      childparent.children[childPosInChildParent] = parent
    end
  end
  fusibles.removeEdge(parent, child)

--  for i=#child.parents,1,-1 do
--    local childparent = child.parents[i]
--    fusibles.addEdge(childparent, parent)
--  end
--  for i=#child.parents,1,-1 do
--    local childparent = child.parents[i]
--    fusibles.removeEdge(childparent, child)
--  end
  return parent
end

