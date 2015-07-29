require 'nngraph'
local ngh = require('nodeGraphHelper')



local x = nn.Identity()()
local n1 = nn.Tanh()(x)
local n2 = nn.Sigmoid()(x)
local n3 = nn.Abs()(n1)
local n4 = nn.CAddTable()({n3, n1})
local n5 = nn.Sigmoid()(n4)
local n6 = nn.Tanh()(n4)
local n7 = nn.CMulTable()({n5, n6})

local g = nn.gModule({x}, {n7})
graph.dot(g.fg, '', 'g')

x = ngh.nnGraphToNgh(g)

local x2 = ngh.walkClone(x)
print('x2=======')
ngh.printGraph(x2)
ngh.dot(x2, '', 'x2')

local r2 = ngh.invertGraph(x2)

print('x=======')
ngh.printGraph(x)
ngh.dot(x, '', 'x')

print('r2=======')
ngh.printGraph(r2)
ngh.dot(r2, '', 'r2')

g = ngh.nghToNnGraph(x)
graph.dot(g.fg, '', 'g.fg')
graph.dot(g.bg, '', 'g.bg')

--local x = nn.Identity()()
--local n1 = nn.Tanh()()
--x.children = {}
--n1.children = {}
--n1.children[x] = 1
--table.insert(n1.children, x)
--print(n1.marked)
--print(x.marked)
--graph.dot(n1:graph(), '', 'n1')

