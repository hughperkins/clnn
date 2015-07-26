require 'graph'
require 'nngraph'

--n1 = graph.Node('one')
--n2 = graph.Node('two')
--n3 = graph.Node('three')
--n4 = graph.Node('four')
--n5 = graph.Node('five')

--n1:add(n2)
--n2:add(n3)
--n2:add(n4)
--n3:add(n4)
--n4:add(n5)

x = nn.Identity()()
n1 = nn.Reshape(2,1)(x)
n2 = nn.SplitTable(2)(n1)
n3, n4 = n2:split(2)
--n1 = nn.Tanh()(x)
--n2 = nn.Sigmoid()(n1)
n5 = nn.Tanh()(n4)
n6 = nn.CAddTable()({n3, n5})
g = nn.gModule({x}, {n6})

x.data.annotations.name = 'x'
n1.data.annotations.name = 'n1'
n2.data.annotations.name = 'n2'
n3.data.annotations.name = 'n3'
n4.data.annotations.name = 'n4'
n5.data.annotations.name = 'n5'
n6.data.annotations.name = 'n6'

function walkGraph(g)
  for i,node in ipairs(g.nodes) do
    children = ''
    for j,child in ipairs(node.children) do
--      children = children .. child.data .. ' '
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

function walkNodes(prefix, node)
  print(prefix, node.data.module)
  for i, child in ipairs(node.children) do
    walkNodes(prefix .. '  ', child)
  end
end

--g = n1:graph()

graph.dot(g.fg, '', 'base.fg') 
graph.dot(g.bg, '', 'base.bg') 

function walkAddParents(node)
  for i, child in ipairs(node.children) do
    child.parents = child.parents or {}
    child.parents[#child.parents + 1] = node
  end
  for i, child in ipairs(node.children) do
    walkAddParents(child)
  end
end

g2 = g:clone()
walkAddParents(g2.innode)
walkAddParents(g2.outnode)
--walkAddParents(g2.fg.nodes[1])
--walkAddParents(g2.bg.nodes[1])
--graphAddParents(g2.fg)
--graphAddParents(g2.bg)
innode = g2.innode.children[1]
innode.children = {}
outnode = g2.outnode.children[1]
print('outnode.parent', outnode.parents[1].data)
print('innode.parent', innode.parents[1].data)
--outnode:roots()[1].parents[1].children = {}
--innode:root()[1].parents[1].children = {}

--innode = outnode:graph():reverse():roots()[1]
if os.getenv('NODE') ~= nil then
  local nodenum = os.getenv('NODE')
  local targetnode = loadstring('return n' .. nodenum)()
  removeNodeByWalk(outnode, targetnode.data)
  --removeNodeFromGraph(g, targetnode.data)
end

--x3 = nngraph.Node(innode.data)
-- x3 is last but one node
-- just walk, and choose last but one...

newbg = g2.bg.nodes[2]
thisnode = newbg
x3 = newbg
while #thisnode.children > 0 do
  x3 = thisnode
  thisnode = thisnode.children[1]
end
print('x3', x3.data.annotations.name)

thisnode = newbg
while thisnode.data ~= x3.data do
  thisnode = thisnode.children[1]
end
thisnode.children = {}

--newbg = newbg.nodes[1].children[1]
--for i, node in ipairs(newbg.nodes)
--end

--out3 = nngraph.Node(outnode.data)
--thisnode = x3
--numnodes = #g2.forwardnodes
--newNodeByData = {}
--newNodeByData[x3.data] = thisnode
--for i, node in ipairs(g2.forwardnodes) do
--  if i > 2 and i < numnodes then
--    print('walk', i, node.data.annotations.name)
--    nextnode = nngraph.Node(node.data)
--    for j, child in ipairs(node.children) do
--    end
--    nextnode:add(thisnode)
--    thisnode = nextnode
----    thisnode = thisnode:add(nextnode)
--    newNodeByData[thisnode] = node.data
--  end
--end

--graph.dot(thisnode:graph(), '', 'n')
--graph.dot(newbg:graph(), '', 'n')

--print('x3.data', x3.data.annotations.name)
--print('newbg.data', newbg.data.annotations.name)
--print('newbg:graph:roots[1].annotations.name', newbg:graph():roots()[1].data.annotations.name)
--print('newbg:graph():reverse():roots[1].annotations.name', newbg:graph():reverse():roots()[1].data.annotations.name)

--x3:add(out3)
g3 = nn.gModule({x3}, {newbg})

for i, node in ipairs(newbg) do
  node.data.mapindex = nil
end

print('innode', innode.id, innode.data.annotations)
print('outnode', outnode.id, outnode.data.annotations)
--innode.data.annotations.name = 'p1'
--outnode.data.annotations.name = 'p2'
--g2 = nn.gModule({innode}, {outnode})

--g.bg = g.outnode:graph()
--g.fg = g.bg:reverse()
----g = n1:graph()

--walkNodes('', n1)
----walkGraph(g)

--for i, node in ipairs(g.forwardnodes) do
--  print(i, node.data.annotations.name, node.data.module)
--end

--graph.dot(innode:graph(), '', 'n') 
--graph.dot(g.bg, '', 'n') 

--graph.dot(innode:graph(), '', 'n') 
graph.dot(g3.fg, '', 'n')

