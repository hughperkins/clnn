require 'nngraph'
require 'clnn'

local fusibletests = {}

function fusibletests.testBasic()
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

  x = nn.Fusible.fromNnGraph(g)

  assert(x:walkValidate())
  local x2 = x:walkClone()

  print('x2=======')
  x2:printGraph()
  x2:dot('', 'x2')

  print('x=======')
  x:printGraph()
  x:dot('', 'x')

  --g = fusibles.anToNnGraph(x)
  --graph.dot(g.fg, '', 'g.fg')
  --graph.dot(g.bg, '', 'g.bg')

  --local x = nn.Identity()()
  --local n1 = nn.Tanh()()
  --x.children = {}
  --n1.children = {}
  --n1.children[x] = 1
  --table.insert(n1.children, x)
  --print(n1.marked)
  --print(x.marked)
  --graph.dot(n1:graph(), '', 'n1')
end

function fusibletests.testSimpleAdd()
  local x = nn.Fusible(1, 'x')
  local n1 = nn.Fusible(1, 'n1')

  x:add(n1)
  x:printGraph()

  if os.getenv('TESTS') ~= nil then x:dot('', 'x') end
  n1 = x.outputs[1].child

  tester:asserteq(#x.inputs, 0)
  tester:asserteq(#x.outputs, 1)
  tester:asserteq(#n1.inputs, 1)
  tester:asserteq(#n1.outputs, 0)
end

function fusibletests.testReduceEdge1()
  local x = nn.Fusible(1, 'x')
  local n1 = nn.Fusible(1, 'n1')

  x:add(n1)
  x:printGraph()

--  local g = nn.gModule({x}, {n1})

--  x = nn.Fusible.fromNnGraph(g)
  if os.getenv('TESTS') ~= nil then x:dot('', 'x') end
  n1 = x.outputs[1].child

  tester:asserteq(#x.inputs, 0)
  tester:asserteq(#x.outputs, 1)
  tester:asserteq(#n1.inputs, 1)
  tester:asserteq(#n1.outputs, 0)

  x = x:reduceEdge(n1)
  tester:asserteq(#x.inputs, 0)
  tester:asserteq(#x.outputs, 0)
end

function fusibletests.testReduceEdgeChildHasChild()
  local x = nn.Fusible(1, 'x')
  local n1 = nn.Fusible(1, 'n1')
  local n2 = nn.Fusible(1, 'n2')

  x:add(n1)
    :add(n2)
  x:printGraph()

--  local g = nn.gModule({x}, {n1})

--  x = nn.Fusible.fromNnGraph(g)
  if os.getenv('TESTS') ~= nil then x:dot('', 'x') end
  n1 = x.outputs[1].child

  tester:asserteq(#x.inputs, 0)
  tester:asserteq(#x.outputs, 1)
  tester:asserteq(#n1.inputs, 1)
  tester:asserteq(#n1.outputs, 1)
  tester:asserteq(#n2.inputs, 1)
  tester:asserteq(#n2.outputs, 0)

  x = x:reduceEdge(n1)
  tester:asserteq(#x.inputs, 0)
  tester:asserteq(#x.outputs, 1)  
  tester:asserteq(#n2.inputs, 1)
  tester:asserteq(#n2.outputs, 0)
end

function go()
  nloop = n_loop or nloop
  local oldtype = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  -- initSeed(seed)
  tester = torch.Tester()
  local targettests = fusibletests
  if os.getenv('LIST') ~= nil then
    print('fusiontests', fusibletests)
    os.exit(0)
  end
  if os.getenv('TESTS') ~= nil then
    targettests = {}
    local filter = os.getenv('TESTS')
    for k, v in pairs(fusibletests) do
      if k:find(filter) ~= nil then
        targettests[k] = v
      end
    end
  end
  print('targettests', targettests)
  tester:add(targettests)
  tester:run(tests)
  torch.setdefaulttensortype(oldtype)
  if print_timing then
    print ''
    print ' ------------------------------------------------------------------------------------------------'
    print '|  Module                                                                          |  Speedup    |'
    print ' ------------------------------------------------------------------------------------------------'
    for module,tm in pairs(times) do
      local str = string.format('| %-80s | %4.2f        |', module, (tm.cpu / (tm.gpu or 1e6)))
      print(str)
    end
    print ' ------------------------------------------------------------------------------------------------'
  end
end

go()


