function fusibles.walkApply(node, func, visited)
  print('node', node)
  visited = visited or {}
  if visited[node] then
    return
  end
  visited[node] = true
  func(node)
  for i, output in ipairs(node.outputs) do
    fusibles.walkApply(output.child, func, visited)
  end
end

function fusibles.printGraph(fusible, prefix, printed)
  printed = printed or {}
  prefix = prefix or ''
  if printed[node] ~= nil then
    return
  end
  printed[fusible] = true
  print(prefix .. tostring(fusible))
  for i, output in ipairs(fusible.outputs) do
    fusibles.printGraph(output.child, prefix .. '  ', printed)
  end
end

function fusibles.printGraphWithLinks(node, prefix, printed)
  printed = printed or {}
  prefix = prefix or ''
  if printed[node] ~= nil then
    return
  end
  printed[node] = true
  print(prefix .. fusibles.nodeToString(node))
  for i, child in ipairs(node.children) do
    print(prefix .. ' - ' .. i .. '->' .. fusibles.getLinkPos(child.parents, node) .. ' ' .. fusibles.nodeToString(child))
  end
  for i, child in ipairs(node.children) do
    fusibles.printGraphWithLinks(child, prefix .. '  ', printed)
  end
end

function fusibles.reverseWalkApply(node, func)
  func(node)
  for i, parent in ipairs(node.parents) do
    fusibles.reverseWalkApply(parent, func)
  end
end

function fusibles.reversePrintGraph(node, prefix, printed)
  printed = printed or {}
  prefix = prefix or ''
  if printed[node] ~= nil then
    return
  end
  printed[node] = true
  print(prefix .. fusibles.nodeToString(node))
  for i, child in ipairs(node.parents) do
    fusibles.reversePrintGraph(child, prefix .. '  ', printed)
  end
end

function fusibles.walkAddReciprocals(nodes)
  fusibles.walkApply(nodes, function(node)
    for i, v in ipairs(node.parents) do
      node.parents[v] = i
    end
    for i, v in ipairs(node.children) do
      node.children[v] = i
    end
  end)
end

function fusibles.count(node)
  local count = 0
  local visited = {}
  fusibles.walkApply(node, function(node)
    if not visited[node] then
      count = count + 1
      visited[node] = true
    end
  end)
  return count
end

function fusibles.reverseCount(node)
  local count = 0
  local visited = {}
  fusibles.reverseWalkApply(node, function(node)
    if not visited[node] then
      count = count + 1
      visited[node] = true
    end
  end)
  return count
end

function fusibles.nodeGraphGetBottom(fusible)
  if #fusible.outputs == 0 then
    return fusible
  end
  return fusibles.nodeGraphGetBottom(fusible.outputs[1].child)
end

function fusibles.nodeGraphGetTop(fusible)
  if #fusible.inputs == 0 then
    return fusible
  end
  return fusibles.nodeGraphGetTop(fusible.inputs[1])
end

