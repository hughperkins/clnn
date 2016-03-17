local _test = clnn._test
local times = _test.times

clnn._testhelpers = {}

function clnn._testhelpers.pointwise_transposed(proto_module, name, max_error)
   max_error = max_error or 1e-7
   local tm = {}
   local title = name .. '.transposed'
   times[title] = tm

   local input = torch.Tensor(11, 19):uniform(-1, 1)
   if name == 'Sqrt' then
      input:uniform(0.1, 1)
   end
   local inputCl = input:clone():cl()

   local cl_module = proto_module:clone():cl()

   -- transpose the inputs and DON'T make contiguous
   input = input:transpose(1, 2)
   inputCl = inputCl:transpose(1, 2)

   local output = proto_module:forward(input)
   local outputCl = cl_module:forward(inputCl)

   local error = outputCl:float() - output
   mytester:assertlt(error:abs():max(), max_error, 'error on state (forward) ')

   local gradOutput = torch.Tensor(11, 19):uniform(-1, 1)
   local gradOutputCl = gradOutput:clone():cl()

   gradOutput = gradOutput:transpose(1, 2)
   gradOutputCl = gradOutputCl:transpose(1, 2)

   local gradInput = proto_module:backward(input, gradOutput)
   local gradInputCl = cl_module:backward(inputCl, gradOutputCl)

   local error = gradInputCl:float() - gradInput
   mytester:assertlt(error:abs():max(), max_error, 'error on state (backward) ')
end

local pointwise_transposed = clnn._testhelpers.pointwise_transposed

