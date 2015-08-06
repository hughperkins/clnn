local Fusible = nn.Fusible

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
--  node.id = fusible.id
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

function Fusible.fromNodes(node)
  -- first it adds all nodes to a list, so 
  -- we can operate on each one on its own
  -- and then we convert each one
  local all_nodes = {}
  fusibles.walkNngraphNodesApply(node, function(node)
    print('fromNodes walk1 node', node.data.module)
    table.insert(all_nodes, node)
  end)

  -- first create fusibles for each node
  local all_fusibles = {}
  local fusible_by_node = {}
  -- walk in reverse order
  for i=#all_nodes, 1, -1 do
    local node = all_nodes[i]
  --for i, node in ipairs(all_nodes) do
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
  for i=#all_nodes, 1, -1 do
    local node = all_nodes[i]
--  for _, node in ipairs(all_nodes) do
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


