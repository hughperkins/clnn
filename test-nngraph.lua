require 'graph'
require 'nngraph'

x = nn.Identity()()
n1 = nn.Reshape(2,1)(x)
n2 = nn.SplitTable(2)(n1)
n3, n4 = n2:split(2)
n5 = nn.Tanh()(n4)
n6 = nn.CAddTable()({n3, n5})
n7 = nn.Sigmoid()(n6)
g = nn.gModule({x}, {n7})

x.data.annotations.name = 'x'
n1.data.annotations.name = 'n1'
n2.data.annotations.name = 'n2'
n3.data.annotations.name = 'n3'
n4.data.annotations.name = 'n4'
n5.data.annotations.name = 'n5'
n6.data.annotations.name = 'n6'
n7.data.annotations.name = 'n7'

function removeNodeByWalk(node, data)
  print('removeNodeByWalk', node.data.annotations.name)
  for i, child in ipairs(node.children) do
    if child.data == data then
      print('remove child', i, child.data.annotations)
      table.remove(node.children, i)
      node.children[child] = nil
      for j, childchild in ipairs(child.children) do
        if node.children[childchild] == nil then
          table.insert(node.children, childchild)
          node.children[childchild] = #node.children
        end
      end
    end
  end
  for i, child in ipairs(node.children) do
    removeNodeByWalk(child, data)
  end
end


-- x3 is last but one node
-- just walk, and choose last but one...

function graphGetNodes(g)
  g2 = g:clone()
  newbg = g2.bg.nodes[2]
  thisnode = newbg
  x = newbg
  while #thisnode.children > 0 do
    x = thisnode
    thisnode = thisnode.children[1]
  end

  thisnode = newbg
  while thisnode.data ~= x.data do
    thisnode = thisnode.children[1]
  end
  thisnode.children = {}

  for i, node in ipairs(newbg) do
    node.data.mapindex = nil
  end

  return x, newbg
end

x, newnodes = graphGetNodes(g)
print('x', torch.type(x))
print('newnodes', torch.type(newnodes))

if os.getenv('NODE') ~= nil then
  local nodenum = os.getenv('NODE')
  local targetname = 'n' .. nodenum
  local targetdata = nil
  for i, node in ipairs(newnodes:graph().nodes) do
    if node.data.annotations.name == targetname then
      targetdata = node.data
      print('got targetdata')
    end
  end
  removeNodeByWalk(newnodes, targetdata)
end

g3 = nn.gModule({x}, {newnodes})

graph.dot(g3.fg, '', 'n')

