require 'nngraph'
local fusibles = require('Fusible')

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

x = fusibles.nnGraphToFusibles(g)

assert(fusibles.walkValidate(x))
local x2 = fusibles.walkClone(x)

print('x2=======')
fusibles.printGraph(x2)
os.exit(0)
fusibles.dot(x2, '', 'x2')

local r2 = fusibles.invertGraph(x2)

print('x=======')
fusibles.printGraph(x)
fusibles.dot(x, '', 'x')

print('r2=======')
fusibles.printGraph(r2)
fusibles.dot(r2, '', 'r2')

g = fusibles.anToNnGraph(x)
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

