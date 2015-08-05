local Fusible = torch.class('nn.Fusible')

Fusibles = {}
fusibles = Fusibles

function Fusible:__init(params)
  self.outputs = {}
  self.inputs = {}
  params = params or {}
  self.numInputs = params.numInputs or 1
  self.numOutputs = params.numOutputs or 1
  self.name = params.name or ''
  self.module = params.module
end

function Fusible:__tostring()
  local res = 'id=' .. tostring(self.id)
  if self.module ~= nil then
    res = res .. ' ' .. tostring(self.module)
  end
  res = res .. ' ' .. tostring(self.numInputs) .. ' -> ' .. tostring(self.numOutputs)
  if self.nSplitOutputs ~= nil then
    res = res .. ' nSplitOutputs=' .. self.nSplitOutputs
  end
  if self.selectindex ~= nil then
    res = res .. ' selectindex=' .. self.selectindex
  end
  return res
end

function Fusible.__concat(a, b)
  return tostring(a) .. tostring(b)
end

-- child operates on output of self
-- assumptions:
-- all outputs from self go to child
function Fusible:add(params)
  if torch.type(params) == 'nn.Fusible' then
    local child = params
    for i=1, self.numOutputs do
      table.insert(child.inputs, self)
      local output = {child=child, outputIdx=i, inputIdx=#child.inputs}
      table.insert(self.outputs, output)
    end
    return child
  else
    local child = nn.Fusible()
    child.numInputs = params.numInputs or 1
    child.numOutputs = params.numOutputs or 1
    child.name = params.name or ''
    child.module = params.module
    for i=1, self.numOutputs do
      table.insert(child.inputs, self)
      table.insert(self.outputs, {child=child, outputIdx=i, inputIdx=1})
    end
    return child
  end
end

-- Fusibles basically take just the 'data' bit of the nnGraph nodes, without
-- the Node container, and without the parents and children of the node
-- this Fusible has:
-- .numInputs (== #.inputs)
-- .numOutputs (number of unique outputIdxs)
-- .outputs = {
--   1 = {outputIdx=1, child=childa, inputIdx=1},
--   2 = {outputIdx=2, child=childb, inputIdx=1},
--   3 = {outputIdx=2, child=childb, inputIdx=2},
-- }
-- .inputs = {
--   1 = parenta
--   2 = parenta
--   3 = parentb
-- }

-- returns x, which is now top of inverted
-- g.bg graph
-- note that datas are identical to the ones
-- in the original graph, not copies/clones
-- I think this probably mutilates the original graph
-- for now (we dont :clone() the original graph,
-- since, this clones the data too, which is
-- not what we want)
function Fusible.fromNnGraph(g)
  local g2 = g
  local newbg = g2.bg.nodes[1]
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

  x = Fusible.fromNodes(newbg)
--  local x = fusibles.invertGraph(newbg)
  x:walkAddDataIds()
--  fusibles.stripNodes(x)
  return x
end

-- pass in a bare, inverted ngh, and
-- receive a gmodule :-)
function fusibles.fusiblesToNnGraph(x)
  local x2 = fusibles.walkClone(x)
  local nodes2 = fusibles.invertGraph(x2)
  fusibles.walkAddBidirectional(nodes2)
  local g = nn.gModule({x2}, {nodes2})
  return g
end

-- create a set of nn.Nodes, using the Fusibles
-- as the data...
function fusibles.walkFusiblesToNodes(fusible, seen)
  seen = seen or {}
  if seen[fusible] ~= nil then
    return seen[fusible]
  end
  local node = nngraph.Node({module=fusible.module, annotations={}})
  node.data.annotations.name = fusible.name
  node.id = fusible.id
  node.data.module = fusible.module
  node.data.selectindex = fusible.selectindex
  node.data.nSplitOutputs = fusible.nSplitOutputs
  seen[fusible] = node
  for i, output in ipairs(fusible.outputs) do
    childNode = fusibles.walkFusiblesToNodes(output.child, seen)
    node:add(childNode, false)
  end
  return node
end

function Fusible.dot(topNode, something, filename)
  local nodes = fusibles.walkFusiblesToNodes(topNode)
  graph.dot(nodes:graph(), something, filename)
--  fusibles.walkRemoveBidirectional(topNode)
end

--function fusibles.stripNodes(x)
--end

-- from http://lua-users.org/wiki/CopyTable
function fusibles.deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolefusibles, etc
        copy = orig
    end
    return copy
end

function fusibles.cloneNode(node)
  local newNode = fusibles.deepcopy(node)
  newNode.module = fusibles.deepcopy(node.module)
  return newNode
end

-- contains a ref to the old module
-- havent decided if this is what we want or not really...
function Fusible.walkClone(fusible, newByOld)
  local newFusible =  nn.Fusible()
  newFusible.module = fusible.module
  newFusible.id = fusible.id
  newFusible.numOutputs = fusible.numOutputs
  newFusible.numInputs = fusible.numInputs
  newFusible.name = fusible.name
  local newByOld = newByOld or {}
  newByOld[fusible] = newFusible
  for i, output in ipairs(fusible.outputs) do
    local oldChild = output.child
    local newChild = newByOld[oldChild]
    if newChild == nil then
      newChild = oldChild:walkClone(newByOld)
    end
    table.insert(newFusible.outputs, {child=newChild,
      outputIdx=output.outputIdx, inputIdx=output.inputIdx})
    local childInputIdx = Fusible.getLinkPos(oldChild.inputs, fusible)
    newChild.inputs[childInputIdx] = newFusible
  end
  return newFusible
end

function fusibles.addNodeLink(from, to, tableName)
  from[tableName] = from[tableName] or {}
  local fromTable = from[tableName]
  if fromTable[to] == nil then
    fromTable[to] = #fromTable + 1
    fromTable[#fromTable + 1] = to
  end
end

function fusibles.addDataLink(from, to, tableName)
  local fromData = from.data
  local toData = to.data
  fromData[tableName] = fromData[tableName] or {}
  local fromTable = fromData[tableName]
  if fromTable[toData] == nil then
    fromTable[toData] = #fromTable + 1
    fromTable[#fromTable + 1] = toData
  end  
end

function Fusible.getName(fusible)
  return fusible.name
end

function Fusible.setName(fusible, name)
  fusible.name = name
end

function Fusible.walkAddDataIds(fusible, dataId)
  dataId = dataId or 0
  if fusible.id == nil then
    dataId = dataId + 1
    fusible.id = dataId
  end
  for i, output in ipairs(fusible.outputs) do
    dataId = Fusible.walkAddDataIds(output.child, dataId)
  end
  return dataId
end

function fusibles.walkReverseAddDataIds(node, dataId)
  dataId = dataId or 0
  if node.data.id == nil then
    dataId = dataId + 1
    node.data.id = dataId
  end
  for i, child in ipairs(node.parents) do
    dataId = fusibles.walkReverseAddDataIds(child, dataId)
  end
  return dataId
end

function fusibles.walkNngraphNodesApply(node, func, visited)
  visited = visited or {}
  if visited[node] then
    return
  end
  visited[node] = true
  func(node)
  for i, child in ipairs(node.children) do
    fusibles.walkNngraphNodesApply(child, func, visited)
  end
end

function fusibles.nodeToString(fusible)
  return tostring(fusible)
end

--function fusibles.walkAddParents(node)
--  node.parents = node.parents or {}
--  for i, child in ipairs(node.children) do
--    child.parents = child.parents or {}
--    fusibles.addLink(child.parents, node)
--    fusibles.walkAddParents(child)
--  end
--end

function Fusible.fromNodes(node)
  -- first it adds all nodes to a list, so 
  -- we can operate on each one on its own
  -- and then we convert each one
  local all_nodes = {}
  fusibles.walkNngraphNodesApply(node, function(node)
    table.insert(all_nodes, node)
  end)

  -- first create fusibles for each node
  local all_fusibles = {}
  local fusible_by_node = {}
  for i, node in ipairs(all_nodes) do
    if fusible_by_node[node] == nil then
      local fusible = nn.Fusible()
      fusible.module = node.data.module
      if fusible.module ~= nil then
        fusible.numInputs = fusible.module.numInputs
        fusible.numOutputs = fusible.module.numOutputs
      end
      if node.data.annotations.name ~= nil then
        fusible.name = node.data.annotations.name
      end
      fusible.selectindex = node.data.selectindex
      fusible.nSplitOutputs = node.data.nSplitOutputs
      fusible_by_node[node] = fusible
      table.insert(all_fusibles, fusible)
--      fusible.id = #all_fusibles
    end
  end

--  for k, v in pairs(fusible_by_node) do
--    print('k', k, 'v', v)
--  end

  -- now, copy the links from nodes
  -- to fusibles
  -- first add outputs to each node.data
  -- that points to the appropriate datas
  -- and also inputs, which points to appropriate
  -- inputs
  -- remember that in the incoming graph, the children are the inputs
  for _, node in ipairs(all_nodes) do
    local data = node.data
    local fusible = fusible_by_node[node]
    for i, child in ipairs(node.children) do
      local childfusible = fusible_by_node[child]
      table.insert(fusible.inputs, childfusible)
      local output = {child=fusible, outputIdx=#childfusible.outputs + 1, inputIdx=#fusible.inputs}
      table.insert(childfusible.outputs, output)
    end
  end

--  local moduleType = torch.type(fusible.module)
  for i, fusible in ipairs(all_fusibles) do
    fusible.numOutputs = 1
    fusible.numInputs = #fusible.inputs
    if fusible.numInputs == 0 then
      fusible.numInputs = 1
    end
  end

--  fusibles.walkAddParents(node)
--  fusibles.walkRemoveBidirectional(node)
  local top = fusible_by_node[node]:getTop()
  top:walkAddDataIds()
  return top
end

-- returns new top
-- cannot do this with Fusible, since inputs and outputs
-- have assymetric information
--function fusibles.invertGraph(top)
--  -- we will put all nodes into all_nodes
--  -- then simply swap the 'children' and 'parents'
--  -- tables.  I guess :-)
--  top = fusibles.nodeGraphGetTop(top)
--  print('top', top)
--  local all_nodes = {}
--  local last_node = nil
--  fusibles.walkApply(top, function(node)
--    if all_nodes[node] == nil then
--      all_nodes[node] = true
--    end
--    last_node = node
--  end)
--  for node, _ in pairs(all_nodes) do
--    local old_parents = node.parents
--    node.parents = node.children
--    node.children = old_parents
--  end
--  return fusibles.nodeGraphGetTop(last_node)
--end

function Fusible.walkValidate(topnode)
  local valid = true
  topnode:walkApply(function(node)
    if node.outputs == nil then
      print('node' .. tostring(node) .. ' has no outputs table')
      valid = false
    end
    if node.inputs == nil then
      print('node' .. tostring(node) .. ' has no inputs table')
      valid = false
    end
    for i, output in ipairs(node.outputs) do
      local child = output.child
      if Fusible.getLinkPos(child.inputs, node) == nil then
        print('child link from ' .. tostring(node) .. ' to ' .. tostring(child) .. ' not reciprocated')
        valid = false
      end
    end
    for i, parent in ipairs(node.inputs) do
      local foundParentOutput = false
      for _, parentoutput in ipairs(parent.outputs) do
        if parentoutput.child == node then
          foundParentOutput = true
        end
      end
      if not foundParentOutput then
        print('parent link from ' .. tostring(node) .. ' to ' .. tostring(parent) .. ' not reciprocated')
        valid = false
      end
    end
  end)
  return valid
end

function Fusible.getLinkPos(targetTable, value)
  for i, v in ipairs(targetTable) do
    if v == value then
      return i
    end
  end
  return nil
end

include 'fusible_walk.lua'
include 'fusible_surgery.lua'

return fusibles

