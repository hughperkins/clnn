nodeGraphHelper = {}

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

function nodeGraphHelper.nameNode(node, name)
  node.data.annotations = node.data.annotations or {}
  node.data.annotations.name = name
end

function nodeGraphHelper.walkAddDataIds(node, dataId)
  dataId = dataId or 1
  node.data.id = dataId
  dataId = dataId + 1
  for i, child in ipairs(node.children) do
    dataId = nodeGraphHelper.walkAddDataIds(child, dataId)
  end
  return dataId
end

function nodeGraphHelper.nodeToString(node)
  local res = tostring(node.data.id)
  if node.data.annotations ~= nil and node.data.annotations.name ~= nil then
    res = res .. ' ' .. node.data.annotations.name
  end
  if node.data.module ~= nil then
    res = res .. ' ' .. torch.type(node.data.module)
  end
  return res
end

function nodeGraphHelper.walkAddParents(node)
  for i, child in ipairs(node.children) do
    nodeGraphHelper.addNodeLink(node, child, 'parents')
  end
end

function nodeGraphHelper.printGraph(node, prefix)
  prefix = prefix or ''
  print(prefix .. nodeGraphHelper.nodeToString(node))
  for i, child in ipairs(node.children) do
    nodeGraphHelper.printGraph(child, prefix .. '  ')
  end
end

function nodeGraphHelper.removeNodeByWalk(node, data)
  print('removeNodeByWalk', node.data.annotations.name)
  if node.data == data then
    -- its me!
    assert(#node.children == 1)
    return node.children[1]
  end
  for i, child in ipairs(node.children) do
    if child.data == data then
      print('remove child', i, child.data.annotations)
      table.remove(node.children, i)
      node.children[child] = nil
      local childmapindexidx = node.data.mapindex[child.data]
      node.data.mapindex[childmapindexidx] = nil
      node.data.mapindex[child.data] = nil
      for j, childchild in ipairs(child.children) do
        if node.children[childchild] == nil then
          table.insert(node.children, childchild)
          node.children[childchild] = #node.children
          node.data.mapindex[childchild.data] = #node.data.mapindex + 1
          node.data.mapindex[#node.data.mapindex + 1] = childchild.data
        end
      end
    end
  end
  for i, child in ipairs(node.children) do
    nodeGraphHelper.removeNodeByWalk(child, data)
  end
  return node
end

return nodeGraphHelper

