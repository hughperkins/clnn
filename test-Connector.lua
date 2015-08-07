require 'nngraph'
require 'clnn'

local mytests = {}

function mytests.testBasic()
  f1 = {name='f1'}
  f2 = {name='f2'}
  e1 = nn.Endpoint(f1)
  e2 = nn.Endpoint(f2)
  connector = nn.Connector({'inner', 'outer'})
  connector.inner:attach(e1)
  connector.outer:attach(e2)
  tester:asserteq(connector.inner:attached(), e1)
  tester:asserteq(connector.outer:attached(), e2)
  tester:asserteq(e1:fixture(), f1)
  tester:asserteq(e1:attached():attached():fixture(), f1)
  tester:asserteq(e1:attached():connected():attached():fixture(), f2)
  tester:asserteq(e2:fixture(), f2)
  tester:asserteq(e2:attached():attached():fixture(), f2)
  tester:asserteq(e2:attached():connected():attached():fixture(), f1)
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


