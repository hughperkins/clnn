--require 'cltorch'
--require 'clnn'
require 'torch'
require 'nn'
require 'sys'

api = os.getenv('API')
if api == nil then
  api = 'cl'
end

if api == 'cl' then
  require 'cltorch'
  require 'clnn'
elseif api == 'cuda' then
  require 'cutorch'
  require 'cunn'
else
  error("unknown api", api)
end

local batchSize = 128

--layer = nn.Tanh()

scenarios = {}
table.insert(scenarios, {name='l1', inplanes=3, insize=128, outplanes=96, filtersize=11}) 
--table.insert(scenarios, {name='l2', inplanes=64, insize=64, outplanes=128, filtersize=9})
table.insert(scenarios, {name='l3', inplanes=128, insize=32, outplanes=128, filtersize=9})
table.insert(scenarios, {name='l4', inplanes=128, insize=16, outplanes=128, filtersize=7})
table.insert(scenarios, {name='l5', inplanes=384, insize=13, outplanes=384, filtersize=3})

--if api == 'cl' then
--  inputcl = input:cl()
--  layercl = layer:cl()
--  layercl:forward(inputcl)
--  for i=1,200 do
--    sys.tic()
--    layercl:forward(inputcl)
--    cltorch.finish()
--    print('sys.toc()', sys.toc())
--  end
--end

function run_scenario(scenario)
  collectgarbage()
  input = torch.Tensor(batchSize, scenario.inplanes, scenario.insize, scenario.insize):uniform()
  layer = nn.SpatialConvolutionMM(scenario.inplanes, scenario.outplanes,
    scenario.filtersize, scenario.filtersize)
  layer.padW = 0
  layer.padH = 0
  if api == 'cl' then
    input = input:cl()
    layer = layer:cl()
  elseif api == 'cuda' then
    input = input:cuda()
    layer = layer:cuda()
  else
    error("unknown api ", api)
  end
  layer:updateOutput(input)
  output = layer.output
  if api == 'cl' then cltorch.finish() end
  if api == 'cuda' then cutorch.synchronize() end
  for i=1,3 do
    sys.tic()
    layer:updateOutput(input)
    if api == 'cl' then cltorch.finish() end
    if api == 'cuda' then cutorch.synchronize() end
    print('   updateOutput sys.toc()', sys.toc())
  end

  -- and now back again, I suppose?
  local backwards_timings = {}
  layer:updateGradInput(input, output)
  if api == 'cl' then cltorch.finish() end
  if api == 'cuda' then cutorch.synchronize() end
  for i=1,3 do
    sys.tic()
    layer:updateGradInput(input, output)
    if api == 'cl' then cltorch.finish() end
    if api == 'cuda' then cutorch.synchronize() end
    local time = sys.toc()
    print('   updateGradInput sys.toc()', time)
    backwards_timings[i] = time
  end

  layer:accGradParameters(input, output)
  if api == 'cl' then cltorch.finish() end
  if api == 'cuda' then cutorch.synchronize() end
  for i=1,3 do
    sys.tic()
    layer:accGradParameters(input, output)
    if api == 'cl' then cltorch.finish() end
    if api == 'cuda' then cutorch.synchronize() end
    time = sys.toc()
    print('   accGradParameters sys.toc()', time)
    backwards_timings[i] = backwards_timings[i] + time
  end
  for i=1,3 do
    print('   backwards', backwards_timings[i])
  end
end

for i, scenario in ipairs(scenarios) do
  print(i, scenario.name)
  run_scenario(scenario)
end


