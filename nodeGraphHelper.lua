nodeGraphHelper = {}
local ngh = nodeGraphHelper

-- returns x, which is now top of inverted
-- g.bg graph
-- note that datas are identical to the ones
-- in the original graph, not copies/clones
-- I think this probably mutilates the original graph
-- for now (we dont :clone() the original graph,
-- since, this clones the data too, which is
-- not what we want)
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
  local x2 = nodeGraphHelper.walkClone(x)
  local nodes2 = nodeGraphHelper.invertGraph(x2)
  nodeGraphHelper.walkAddBidirectional(nodes2)
  local g = nn.gModule({x2}, {nodes2})
  return g
end

function nodeGraphHelper.dot(topNode, something, filename)
--  if torch.type(topNodes) ~= 'table' then
--    topNodes = {topNodes}
--  end
--  local topNode = nn.Identity()()
--  ngh.nodeSetName(topNode, 'topNode')
--  for i=1, #topNodes do
  local topNodes = nodeGraphHelper.walkClone(topNode)
--    ngh.addEdge(topNode, nghclone)
--  end
  nodeGraphHelper.walkAddBidirectional(topNode)
  graph.dot(topNode:graph(), something, filename)
  nodeGraphHelper.walkRemoveBidirectional(topNode)
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
    child.parents = child.parents or {}
    nodeGraphHelper.addLink(child.parents, node)
    nodeGraphHelper.walkAddParents(child)
  end
end

function nodeGraphHelper.walkRemoveBidirectional(node)
  nodeGraphHelper.walkApply(node, function(node)
    node.data.mapindex = nil
    for k,v in pairs(node.children) do
      if torch.type(k) ~= 'number' then
        node.children[k] = nil
      end
    end
    for k,v in pairs(node.parents) do
      if torch.type(k) ~= 'number' then
        node.parents[k] = nil
      end
    end
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
  top = nodeGraphHelper.nodeGraphGetTop(top)
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
end

function ngh.walkValidate(topnode)
  local valid = true
  ngh.walkApply(topnode, function(node)
    if node.children == nil then
      print('node' .. tostring(node) .. ' has no children table')
      valid = false
    end
    if node.parents == nil then
      print('node' .. tostring(node) .. ' has no parents table')
      valid = false
    end
    if node.data.mapindex ~= nil then
      print('node' .. tostring(node) .. ' has mapindex table')
      assert(node.data.mapindex == nil)
      valid = false
    end
    for k, v in pairs(node.children) do
      if torch.type(k) ~= 'number' then
        print('node' .. tostring(node) .. ' has non-number children key')
        assert(torch.type(k) == 'number')
        valid = false
      end
    end
    for k, v in pairs(node.parents) do
      if torch.type(k) ~= 'number' then
        print('node' .. tostring(node) .. ' has non-number parents key')
        assert(torch.type(k) == 'number')
        valid = false
      end
    end
    for i, child in ipairs(node.children) do
      if ngh.getLinkPos(child.parents, node) == nil then
        print('child link from ' .. tostring(node) .. ' to ' .. tostring(child) .. ' not reciprocated')
        valid = false
      end
    end
    for i, parent in ipairs(node.parents) do
      if ngh.getLinkPos(parent.children, node) == nil then
        print('parent link from ' .. tostring(node) .. ' to ' .. tostring(parent) .. ' not reciprocated')
        valid = false
      end
    end
  end)
  return valid
end

function nodeGraphHelper.removeLink(targetTable, value)
  local pos = nodeGraphHelper.getLinkPos(targetTable, value)
  if pos ~= nil then
    table.remove(targetTable, pos)
  end
end

function nodeGraphHelper.addEdge(parent, child)
  nodeGraphHelper.addLink(parent.children, child)
  nodeGraphHelper.addLink(child.parents, parent)
end

function nodeGraphHelper.reduceEdge(parent, child)
  nodeGraphHelper.removeLink(parent.children, child)
  for i, childchild in ipairs(child.children) do
    nodeGraphHelper.addLink(parent.children, childchild)
    nodeGraphHelper.removeLink(childchild.parents, child)
    nodeGraphHelper.addLink(childchild.parents, parent)
  end
  -- also remove any parents from child which are same as parent's parents
  for i, parentparent in ipairs(parent.parents) do
    ngh.removeLink(child.parents, parentparent)
    ngh.removeLink(parentparent.children, child)
  end
--  for i, childparent in ipairs(child.parents) do
--    parentparentpos = ngh.getLinkPos(parent.parents, childparent)
--    if parentparentpos ~= nil then
--      ngh.removeLink(child.parents, 
--    end
--  end
  return parent
end

return nodeGraphHelper

