local Fusible = nn.Fusible

function Fusible.walkApply(fusible, func, visited)
  print('fusible', fusible)
  visited = visited or {}
  if visited[fusible] then
    return
  end
  visited[fusible] = true
  func(fusible)
  for i, output in ipairs(fusible.outputs) do
    Fusible.walkApply(output.child, func, visited)
  end
end

function Fusible.printGraph(fusible, prefix, printed)
  printed = printed or {}
  prefix = prefix or ''
  if printed[fusible] ~= nil then
    return
  end
  printed[fusible] = true
  print(prefix .. tostring(fusible))
  for i, output in ipairs(fusible.outputs) do
    Fusible.printGraph(output.child, prefix .. '  ', printed)
  end
end

function Fusible.printGraphWithLinks(fusible, prefix, printed)
  printed = printed or {}
  prefix = prefix or ''
  if printed[fusible] ~= nil then
    return
  end
  printed[fusible] = true
  print(prefix .. tostring(fusible))
  for i, output in ipairs(fusible.outputs) do
    local child = output.child
    print(prefix .. ' - ' .. i .. '->' .. Fusible.getLinkPos(child.inputs, fusible) .. ' ' .. tostring(child))
  end
  for i, output in ipairs(fusible.outputs) do
    Fusible.printGraphWithLinks(output.child, prefix .. '  ', printed)
  end
end

function Fusible.reverseWalkApply(fusible, func)
  func(fusible)
  for i, input in ipairs(fusible.inputs) do
    Fusible.reverseWalkApply(input, func)
  end
end

function Fusible.reversePrintGraph(fusible, prefix, printed)
  printed = printed or {}
  prefix = prefix or ''
  if printed[fusible] ~= nil then
    return
  end
  printed[fusible] = true
  print(prefix .. Fusible.fusibleToString(fusible))
  for i, input in ipairs(fusible.inputs) do
    Fusible.reversePrintGraph(input, prefix .. '  ', printed)
  end
end

--function Fusible.walkAddReciprocals(fusibles)
--  Fusible.walkApply(fusibles, function(fusible)
--    for i, v in ipairs(fusible.parents) do
--      fusible.parents[v] = i
--    end
--    for i, v in ipairs(fusible.children) do
--      fusible.children[v] = i
--    end
--  end)
--end

function Fusible.count(fusible)
  local count = 0
  local visited = {}
  Fusible.walkApply(fusible, function(fusible)
    if not visited[fusible] then
      count = count + 1
      visited[fusible] = true
    end
  end)
  return count
end

function Fusible.reverseCount(fusible)
  local count = 0
  local visited = {}
  Fusible.reverseWalkApply(fusible, function(fusible)
    if not visited[fusible] then
      count = count + 1
      visited[fusible] = true
    end
  end)
  return count
end

function Fusible.getBottom(fusible)
  if #fusible.outputs == 0 then
    return fusible
  end
  return Fusible.getBottom(fusible.outputs[1].child)
end

function Fusible.getTop(fusible)
  if #fusible.inputs == 0 then
    return fusible
  end
  return Fusible.getTop(fusible.inputs[1])
end

