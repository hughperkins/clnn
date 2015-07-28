nodeGraphHelper = {}

-- returns x, which is now top of inverted
-- g.bg graph
-- note that datas are identical to the ones
-- in the original graph, not copies/clones
function nodeGraphHelper.nnGraphToNgh(g)
  local g2 = g
  local newbg = g2.bg.nodes[2]
  local thisnode = newbg
  local x = newbg
  while #thisnode.children > 0 do
    x = thisnode
    thisnode = thisnode.children[1]
  end

  thisnode = newbg
  while thisnode.data ~= x.data do
    thisnode = thisnode.children[1]
  end
  thisnode.children = {}

  nodeGraphHelper.walkStandardize(newbg)
  local x = nodeGraphHelper.invertGraph(newbg)
  nodeGraphHelper.walkAddDataIds(x)
  return x
end

-- pass in a bare, inverted ngh, and
-- receive a gmodule :-)
function nodeGraphHelper.nghToNnGraph(x)
  local nodes = nodeGraphHelper.invertGraph(x)
--  nodeGraphHelper.
  local g = nn.gModule({x}, {nodes})
  return g
end

function nodeGraphHelper.dot(topNode, something, filename)
  local nghclone = nodeGraphHelper.walkClone(topNode)
  nodeGraphHelper.walkAddBidirectional(nghclone)
  graph.dot(nghclone:graph(), something, filename)
end

function nodeGraphHelper.walkClone(node, newByOld)
  local newGraph = nngraph.Node(node.data)
  newGraph.parents = {}
  local newByOld = newByOld or {}
  newByOld[node] = newGraph
  for i, oldChild in ipairs(node.children) do
    local newChild = newByOld[oldChild]
    if newChild == nil then
      newChild = nodeGraphHelper.walkClone(oldChild, newByOld)
    end
    table.insert(newGraph.children, newChild)
    table.insert(newChild.parents, newGraph)
  end
  return newGraph
end

function nodeGraphHelper.addNodeLink(from, to, tableName)
  from[tableName] = from[tableName] or {}
  local fromTable = from[tableName]
  if fromTable[to] == nil then
    fromTable[to] = #fromTable + 1
    fromTable[#fromTable + 1] = to
  end
end

function nodeGraphHelper.addDataLink(from, to, tableName)
  local fromData = from.data
  local toData = to.data
  fromData[tableName] = fromData[tableName] or {}
  local fromTable = fromData[tableName]
  if fromTable[toData] == nil then
    fromTable[toData] = #fromTable + 1
    fromTable[#fromTable + 1] = toData
  end  
end

function nodeGraphHelper.nodeGetName(node)
  if node.data.annotations == nil then
    return nil
  end
  return node.data.annotations.name
end

function nodeGraphHelper.nodeSetName(node, name)
  nodeGraphHelper.nameNode(node, name)
end

function nodeGraphHelper.nameNode(node, name)
  node.data.annotations = node.data.annotations or {}
  node.data.annotations.name = name
end

function nodeGraphHelper.walkAddDataIds(node, dataId)
  dataId = dataId or 0
  if node.data.id == nil then
    dataId = dataId + 1
    node.data.id = dataId
  end
  for i, child in ipairs(node.children) do
    dataId = nodeGraphHelper.walkAddDataIds(child, dataId)
  end
  return dataId
end

function nodeGraphHelper.walkReverseAddDataIds(node, dataId)
  dataId = dataId or 0
  if node.data.id == nil then
    dataId = dataId + 1
    node.data.id = dataId
  end
  for i, child in ipairs(node.parents) do
    dataId = nodeGraphHelper.walkReverseAddDataIds(child, dataId)
  end
  return dataId
end

function nodeGraphHelper.walkApply(node, func)
  func(node)
  for i, child in ipairs(node.children) do
    nodeGraphHelper.walkApply(child, func)
  end
end

function nodeGraphHelper.nodeToString(node)
  local res = tostring(node.data.id)
  if node.data.annotations ~= nil and node.data.annotations.name ~= nil then
    res = res .. ' ' .. node.data.annotations.name
  end
  if node.data.module ~= nil then
    res = res .. ' ' .. tostring(node.data.module)
  end
  return res
end

function nodeGraphHelper.count(node)
  local count = 0
  local visited = {}
  nodeGraphHelper.walkApply(node, function(node)
    if not visited[node] then
      count = count + 1
      visited[node] = true
    end
  end)
  return count
end

function nodeGraphHelper.reverseCount(node)
  local count = 0
  local visited = {}
  nodeGraphHelper.reverseWalkApply(node, function(node)
    if not visited[node] then
      count = count + 1
      visited[node] = true
    end
  end)
  return count
end

function nodeGraphHelper.nodeGraphGetBottom(node)
  if #node.children == 0 then
    return node
  end
  return nodeGraphHelper.nodeGraphGetBottom(node.children[1])
end

function nodeGraphHelper.nodeGraphGetTop(node)
  if #node.parents == 0 then
    return node
  end
  return nodeGraphHelper.nodeGraphGetTop(node.parents[1])
end

function nodeGraphHelper.walkAddParents(node)
  node.parents = node.parents or {}
  for i, child in ipairs(node.children) do
--    nodeGraphHelper.addNodeLink(child, node, 'parents')
    child.parents = child.parents or {}
    nodeGraphHelper.addLink(child.parents, node)
    nodeGraphHelper.walkAddParents(child)
  end
end

function nodeGraphHelper.walkRemoveBidirectional(node)
  nodeGraphHelper.walkApply(node, function(node)
    node.data.mapindex = nil
    for k,v in pairs(node.children) do
      if torch.type(k) == 'nn.Node' then
        node.children[k] = nil
      end
    end
--    for k,v in pairs(node.parents) do
--      if torch.type(k) == 'nn.Node' then
--        node.parents[k] = nil
--      end
--    end
  end)
end

function nodeGraphHelper.walkAddBidirectional(node)
  nodeGraphHelper.walkApply(node, function(node)
    node.data.mapindex = {}
    for i,v in ipairs(node.children) do
      node.children[v] = i
      node.data.mapindex[v.data] = i
      node.data.mapindex[i] = v.data
    end
  end)
end

function nodeGraphHelper.walkStandardize(node)
  nodeGraphHelper.walkAddParents(node)
  nodeGraphHelper.walkRemoveBidirectional(node)
end

function nodeGraphHelper.printGraph(node, prefix, printed)
  printed = printed or {}
  prefix = prefix or ''
  if printed[node] ~= nil then
    return
  end
  printed[node] = true
  print(prefix .. nodeGraphHelper.nodeToString(node))
  for i, child in ipairs(node.children) do
    nodeGraphHelper.printGraph(child, prefix .. '  ', printed)
  end
end

function nodeGraphHelper.reverseWalkApply(node, func)
  func(node)
  for i, parent in ipairs(node.parents) do
    nodeGraphHelper.reverseWalkApply(parent, func)
  end
end

function nodeGraphHelper.reversePrintGraph(node, prefix, printed)
  printed = printed or {}
  prefix = prefix or ''
  if printed[node] ~= nil then
    return
  end
  printed[node] = true
  print(prefix .. nodeGraphHelper.nodeToString(node))
  for i, child in ipairs(node.parents) do
    nodeGraphHelper.reversePrintGraph(child, prefix .. '  ', printed)
  end
end

function nodeGraphHelper.walkAddReciprocals(nodes)
  nodeGraphHelper.walkApply(nodes, function(node)
    for i, v in ipairs(node.parents) do
      node.parents[v] = i
    end
    for i, v in ipairs(node.children) do
      node.children[v] = i
    end
  end)
end

function nodeGraphHelper.getLinkPos(targetTable, value)
  for i, v in ipairs(targetTable) do
    if v == value then
      return i
    end
  end
end

-- returns new top
function nodeGraphHelper.invertGraph(top)
  -- we will put all nodes into all_nodes
  -- then simply swap the 'children' and 'parents'
  -- tables.  I guess :-)
  local all_nodes = {}
  local last_node = nil
  nodeGraphHelper.walkApply(top, function(node)
    if all_nodes[node] == nil then
      all_nodes[node] = true
    end
    last_node = node
  end)
  for node, _ in pairs(all_nodes) do
    local old_parents = node.parents
    node.parents = node.children
    node.children = old_parents
  end
  return nodeGraphHelper.nodeGraphGetTop(last_node)
end

function nodeGraphHelper.addLink(targetTable, value)
  if nodeGraphHelper.getLinkPos(targetTable, value) == nil then
    table.insert(targetTable, value)
  end
--  targetTable[value] = #targetTable
end

function nodeGraphHelper.removeLink(targetTable, value)
  table.remove(targetTable, nodeGraphHelper.getLinkPos(targetTable, value))
end

function nodeGraphHelper.addEdge(parent, child)
  nodeGraphHelper.addLink(parent.children, child)
  nodeGraphHeler.addLink(child.parents, parent)
end

function nodeGraphHelper.reduceEdge(parent, child)
  -- means:
  -- - all childs children transfer to parent
  nodeGraphHelper.removeLink(parent.children, child)
--  print('parent', torch.type(parent), parent.data.module, parent.data.id, torch.type(parent.parents))
  for i, childchild in ipairs(child.children) do
--    print('child', torch.type(child), child.data.module, child.data.id, torch.type(child.parents))
--    print('childchild', torch.type(childchild), childchild.data.module, childchild.data.id, torch.type(childchild.parents))
    nodeGraphHelper.addLink(parent.children, childchild)
    nodeGraphHelper.removeLink(childchild.parents, child)
    nodeGraphHelper.addLink(childchild.parents, parent)
  end
  return parent
end

-- pass in the node we want to remove
-- must have set parents beforehand
-- joins from and to together, removing an edge
-- from the graph
-- the resulting data will be that in parent
-- that in child will be thrown away
--function nodeGraphHelper.fuseNodes(parent, child)
--  -- remove parent from child
--  table.remove(child.parents, child.parents[parent])
--  child.parents[parent] = nil
--  for i, child in to.children do
--    
--  end
--end

--function nodeGraphHelper.removeNodeByWalk(node, data)
--  print('removeNodeByWalk', node.data.annotations.name)
--  if node.data == data then
--    print('its me!')
--    assert(#node.children == 1)
--    return node.children[1]
--  end
--  for i, child in ipairs(node.children) do
--    if child.data == data then
--      print('remove child', i, child.data.annotations.name)
--      table.remove(node.children, i)
--      node.children[child] = nil
--      table.remove(node.data.mapindex, node.data.mapindex[child.data])
--      node.data.mapindex[child.data] = nil

--      

--      for j, childchild in ipairs(child.children) do
--        if node.children[childchild] == nil then
--          table.insert(node.children, childchild)
--          node.children[childchild] = #node.children
--          node.data.mapindex[childchild.data] = #node.data.mapindex + 1
--          node.data.mapindex[#node.data.mapindex + 1] = childchild.data
--        end
--      end
--    end
--  end
--  for i, child in ipairs(node.children) do
--    child = nodeGraphHelper.removeNodeByWalk(child, data)
--  end
--  return node
--end

return nodeGraphHelper

