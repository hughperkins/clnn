graphHelper = {}

function graphHelper.graphGetNodes(g)
--  g2 = g:clone()
  g2 = g
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

  for i, node in ipairs(newbg:graph().nodes) do
    node.data.mapindex = {}
    for j, child in ipairs(node.children) do
      node.data.mapindex[#node.data.mapindex + 1] = child.data
      node.data.mapindex[child.data] = #node.data.mapindex
    end
  end

  return x, newbg
end

return graphHelper

