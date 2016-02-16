local _test = clnn._test
local times = _test.times
local clnntest = _test.clnntest
local x_clnntest = _test.x_clnntest
local nloop = _test.nloop
local precision_forward = 0.01
local precision_backward = 0.01

function clnntest.LookupTable_forward()
   local nVocab = 10000
   local nDim = 100
   local nInput = 1000

   local tm = {}
   local title = string.format('LookupTable forward %d x %d', nVocab, nDim)
   times[title] = tm

   local input = torch.LongTensor(nInput):random(nVocab)
   local sconv = nn.LookupTable(nVocab, nDim)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
       groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cl()
   local gconv = sconv:cl()
   local resopencl = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
       resopencl = gconv:forward(input)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local error = resopencl:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state')
end

function clnntest.LookupTable_backward()
   local nVocab = 10000

   for _,nDim in ipairs{97,255} do
      for _,nInput in ipairs{10,101} do
--      for _,nInput in ipairs{10,101,1000,10007} do
--         for _,scaleGradByFreq in ipairs{false,true} do
            for _,batch in ipairs{false, true} do
               local input, gradOutput
               if batch then
                  input = torch.LongTensor(nInput, 5):random(nVocab)
                  gradOutput = torch.randn(nInput, 5, nDim)
               else
                  input = torch.LongTensor(nInput):random(nVocab)
                  gradOutput = torch.randn(nInput, nDim)
               end

               local sconv = nn.LookupTable(nVocab, nDim)
               local gconv = sconv:clone():cl()
--               if scaleGradByFreq then
--                  sconv = sconv:scaleGradByFreq()
--                  gconv = gconv:scaleGradByFreq()
--               end

               sconv:forward(input)
               sconv:backward(input, gradOutput)

               input = input:cl()
               gradOutput = gradOutput:cl()
               gconv:forward(input)
               gconv:backward(input, gradOutput)

               local weightGradError = gconv.gradWeight:float() - sconv.gradWeight
               print('nDim', nDim, 'nInput', nInput, 'batch', batch, 'error', weightGradError:abs():max())
               mytester:assertlt(weightGradError:abs():max(), precision_backward,
                  'error on weight for size ' .. tostring(nInput) .. ' scaleGradByFreq: ' .. tostring(scaleGradByFreq)
                  .. ' nDim ' .. tostring(nDim))
            end
--         end
      end
   end

   local nDim = 128
   local nInput = 1000
   local tm = {}
   local title = string.format('LookupTable backward %d x %d', nVocab, nDim, nInput)
   times[title] = tm

   local input = torch.LongTensor(nInput):random(nVocab)
   local gradOutput = torch.randn(nInput, nDim)
   local sconv = nn.LookupTable(nVocab, nDim)
   local gconv = sconv:clone():cl()

   sconv:forward(input)
   sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
       sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cl()
   gradOutput = gradOutput:cl()
   gconv:forward(input)
   gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
       gconv:backward(input, gradOutput)
   end
   cltorch.synchronize()
   tm.gpu = a:time().real

   local weightGradError = gconv.gradWeight:float() - sconv.gradWeight
   mytester:assertlt(weightGradError:abs():max(), precision_backward, 'error on weight')
end

