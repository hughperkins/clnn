require 'nngraph'
require 'clnn'

local mytests = {}

function mytests.testBasic()
  f1 = {name='f1'}
  f2 = {name='f2'}
  a = nn.Endpoint(f1)
  b = nn.Endpoint(f2)
  tester:asserteq(a.fixture, f1)
  tester:asserteq(b.fixture, f2)
  tester:asserteq(a:other(), nil)
  tester:asserteq(b:other(), nil)
  tester:asserteq(a:numAttached(), 0)
  tester:asserteq(b:numAttached(), 0)
  a:attach(b)
  tester:asserteq(a:numAttached(), 1)
  tester:asserteq(b:numAttached(), 1)
  tester:asserteq(a:other(), b)
  tester:asserteq(b:other(), a)
  b:detach(a)
  tester:asserteq(a:other(), nil)
  tester:asserteq(b:other(), nil)
  tester:asserteq(a:numAttached(), 0)
  tester:asserteq(b:numAttached(), 0)
end

function go()
  nloop = n_loop or nloop
  local oldtype = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  -- initSeed(seed)
  tester = torch.Tester()
  local targettests = mytests
  if os.getenv('LIST') ~= nil then
    print('tests', tests)
    os.exit(0)
  end
  if os.getenv('TESTS') ~= nil then
    targettests = {}
    local filter = os.getenv('TESTS')
    for k, v in pairs(tests) do
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


