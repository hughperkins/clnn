require 'graph'
require 'nngraph'

n1 = graph.Node('one')
n2 = graph.Node('two')
n3 = graph.Node('three')
n4 = graph.Node('four')
n5 = graph.Node('five')

n1:add(n2)
n2:add(n3)
n2:add(n4)
n3:add(n4)
n4:add(n5)

function walkGraph(g)
  for i,node in ipairs(g.nodes) do
    children = ''
    for j,child in ipairs(node.children) do
      children = children .. child.data .. ' '
    end
    print(i, node.data, ':', children)
  end
end

function removeNodeByWalk(node, data)
  print('removeNodeByWalk', node.data)
  for i, child in ipairs(node.children) do
    if child.data == data then
      print('remove child', i, child.data)
      table.remove(node.children, i)
      node.children[child] = nil
      for j, childchild in ipairs(child.children) do
        if node.children[childchild] == nil then
          table.insert(node.children, childchild)
          node.children[childchild] = #node.children
        end
      end
      -- child.children = {}
--      return
    end
  end
  for i, child in ipairs(node.children) do
    removeNodeByWalk(child, data)
  end
end

function indexGraph(g)
  g.edgesByData = {}
  for i, edge in ipairs(g.edges) do
    g.edgesByData[edge.from.data] = g.edgesByData[edge.from.data] or {}
    local fromList = g.edgesByData[edge.from.data]
    fromList[#fromList + 1] = edge
    g.edgesByData[edge.to.data] = g.edgesByData[edge.to.data] or {}
    local toList = g.edgesByData[edge.to.data]
    toList[#toList + 1] = edge
  end
end

function removeEdge(g, data)
  for i, edge in ipairs(g.edges) do
--    if edge.from.data == data or edge.to.
  end
end

function removeNodeFromWalkGraph(g, node, data)
  print('removeNodeFromWalkGraph', node.data)
  for i, child in ipairs(node.children) do
    if child.data == data then
      print('remove child', i, child.data)
      table.remove(node.children, i)
      node.children[child] = nil
      for j, childchild in ipairs(child.children) do
        if node.children[childchild] == nil then
          table.insert(node.children, childchild)
          node.children[childchild] = #node.children
        end
      end
      -- child.children = {}
--      return
    end
  end
  for i, child in ipairs(node.children) do
    removeNodeFromWalkGraph(child, data)
  end  
end

function removeNodeFromGraph(g, data)
--  removeEdge(g, data)
  if g.edgesByData == nil then
    indexGraph(g)
  end
  local edges = g.edgesByData[data]
  for i, edge in ipairs(edges) do
    print(i, edge.from.data, edge.to.data, g.edges[edge])
    
  end
  removeNodeByWalk(g.nodes[1], data)
--  for i,node in ipairs(g.nodes) do
--    if node.data == data then
--      table.remove(no
--      for j,child in ipairs(node) do
--        node.children[#node.children+1] = child
--      end
--    end
--  end
end

function walkNodes(prefix, node)
  print(prefix .. node.data, node.marked, node.visited)
  for i, child in ipairs(node.children) do
    walkNodes(prefix .. '  ', child)
  end
end

--local posToRemove = nil
--for i,child in ipairs(n1.children) do
--  if child.data == n2.data then
--    posToRemove = i
----    table.remove(n1.children, posToRemove)
--  end
--end
--for i,childchild in ipairs(n2) do
--  n1.children[#n1.children + 1] = childchild
--end
--table.remove(n2.children, posToRemove)

--g = n1:graph()
--walkGraph(g)
g = n1:graph()

if os.getenv('NODE') ~= nil then
  local nodenum = os.getenv('NODE')
  local targetnode = loadstring('return n' .. nodenum)()
  removeNodeByWalk(n1, targetnode.data)
  --removeNodeFromGraph(g, targetnode.data)
end
g = n1:graph()

walkNodes('', n1)
--print('n1.chlidren', #n1.children)
--for k,v in pairs(n1.children) do
--  print('k,v', 'k.data', 'k==nil', k == nil, 'v', v, 'type k', torch.type(k), 'type v', torch.type(v)) -- v.data)
--  if torch.type(k) == 'graph.Node' then
--    print('k.data', k.data)
--  end
--end
--walkNodes('', n1)
--print('n1.chlidren', #n1.children)
--g2 = n1:graph()
--walkNodes('', n1)
--print('n1.chlidren', #n1.children)
if true then
--  print('n1.chlidren', #n1.children)
  --removeNodeFromGraph(g, n2.data)
  walkGraph(g)
  for i, edge in ipairs(g.edges) do
    print('edge', i, edge.from.data, edge.to.data)
  end

--  print('#n2.children', #n2.children)

--  for i, node in ipairs(g.nodes) do
--    print('i', i, node, #node.children)
--  end

  graph.dot(g, '', 'n') 
end

