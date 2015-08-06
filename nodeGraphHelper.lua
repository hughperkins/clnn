-- operations on nnGraph.Node

nngraph.nodeGraphHelper = {}
local ngh = nngraph.nodeGraphHelper

function ngh.walkApply(node, func, seen)
  if seen == nil then
    print('')
  end
  print('ngh.walkApply node ', torch.type(node.data.module))
  seen = seen or {}
  if seen[node] then
    return
  end
  seen[node] = true
  func(node)
  for i, child in ipairs(node.children) do
    ngh.walkApply(child, func, seen)
  end
end

function ngh.addParents(node)
  ngh.walkApply(node, function(node)
    print('addParents node', torch.type(node.data.module))
    node.parents = node.parents or {}
    for i, child in ipairs(node.children) do
      child.parents = child.parents or {}
      table.insert(child.parents, node)
    end
  end)
end

function ngh.removeParents(node)
  node = ngh.top(node)
  ngh.walkApply(node, function(node)
    node.parents = nil
  end)
end

-- must have added parents first
function ngh.invert(node)
  -- add nodes to list, then swap parents <-> children
  local all_nodes = {}
  ngh.walkApply(node, function(node)
    table.insert(all_nodes, node)
  end)
  
  for i, node in ipairs(all_nodes) do
    local old_parents = node.parents
    node.parents = node.children
    node.children = old_parents
  end
  return ngh.top(node)
end

function ngh.top(node)
  if #node.parents == 0 then
    return node
  end
  return ngh.top(node.parents[1])
end

function ngh.bottom(node)
  if #node.children == 0 then
    return node
  end
  return ngh.top(node.children[1])
end

