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
table.insert(scenarios, {name='l2', inplanes=64, insize=64, outplanes=128, filtersize=9})
table.insert(scenarios, {name='l3', inplanes=128, insize=32, outplanes=128, filtersize=9})
table.insert(scenarios, {name='l4', inplanes=128, insize=16, outplanes=128, filtersize=7})
table.insert(scenarios, {name='l5', inplanes=384, insize=13, outplanes=384, filtersize=3})

--if api == 'cl' then
--   inputcl = input:cl()
--   layercl = layer:cl()
--   layercl:forward(inputcl)
--   for i=1,200 do
--      sys.tic()
--      layercl:forward(inputcl)
--      cltorch.finish()
--      print('sys.toc()', sys.toc())
--   end
--end

function run_scenario(scenario, numIts)
   collectgarbage()
   print('scenario ' .. scenario.name .. ' inplanes=' .. scenario.inplanes .. ' insize=' ..scenario.insize .. ' filtersize=' .. scenario.filtersize)
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
   local sumForward = 0
   for i=1,numIts do
      sys.tic()
      layer:updateOutput(input)
      if api == 'cl' then cltorch.finish() end
      if api == 'cuda' then cutorch.synchronize() end
      local thisTime = sys.toc() * 1000
      print('    updateOutput ' .. thisTime .. 'ms')
      sumForward = sumForward + thisTime
   end
   local averageForward = sumForward / numIts

   -- and now back again
   layer:updateGradInput(input, output)
   if api == 'cl' then cltorch.finish() end
   if api == 'cuda' then cutorch.synchronize() end
   local sumGradInput = 0
   for i=1,numIts do
      sys.tic()
      layer:updateGradInput(input, output)
      if api == 'cl' then cltorch.finish() end
      if api == 'cuda' then cutorch.synchronize() end
      local thisTime = sys.toc() * 1000
      print('    updateGradInput ' .. thisTime .. 'ms')
      sumGradInput = sumGradInput + thisTime
   end
   local averageGradInput = sumGradInput / numIts

   layer:accGradParameters(input, output)
   if api == 'cl' then cltorch.finish() end
   if api == 'cuda' then cutorch.synchronize() end
   local sumUpdateWeights = 0
   for i=1,numIts do
      sys.tic()
      layer:accGradParameters(input, output)
      if api == 'cl' then cltorch.finish() end
      if api == 'cuda' then cutorch.synchronize() end
      thisTime = sys.toc() * 1000
      print('    accGradParameters ' .. thisTime .. 'ms')
      sumUpdateWeights = sumUpdateWeights + thisTime
   end
   local averageUpdateWeights = sumUpdateWeights / numIts
   print('Average forward:', averageForward .. 'ms')
   print('Average updateGradInput:', averageGradInput .. 'ms')
   print('Average updateWeights:', averageUpdateWeights .. 'ms')
   print('Average backward:', (averageGradInput + averageUpdateWeights) .. 'ms')
end

local numIts = 10
if os.getenv('NUMITS') then
   numIts = tonumber(os.getenv('NUMITS'))
end
print('Number iterations: ' .. numIts)
for i, scenario in ipairs(scenarios) do
   if os.getenv('NET') == nil or scenario.name == os.getenv('NET') then
      run_scenario(scenario, numIts)
   end
end

