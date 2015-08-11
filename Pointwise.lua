local definitions = {
   {
      name = 'Sigmoid',
      forward_op = 'output = 1.f / (1.f + exp( - input))',
      backward_op = 'gradInput = gradOutput * output * (1.f - output)'
   },
   {
      name = 'Sqrt',
      forward_op = 'output = sqrt(input)',
      backward_op = 'gradInput = (output == 0.0f) ? 0.0f : ((0.5f * gradOutput) / output)'
   },
   {
      name = 'Square',
      forward_op = 'output = input * input',
      backward_op = 'gradInput = 2.0f * gradOutput * input'
   },
   {
      name = 'Exp',
      forward_op = 'output = exp(input)',
      backward_op = 'gradInput = gradOutput * output'
   },
   {
      name = 'Abs',
      forward_op = 'output = fabs(input)',
      backward_op = 'gradInput = input < 0 ? - gradOutput : gradOutput'
   },
   {
      name = 'LogSigmoid',
      forward_op = 'output = -log(1.f + exp(-input))',
      backward_op = 'float v = exp(-input); gradInput = gradOutput * v / (1.f + v)'
   },
   {
      name = 'HardTanh',
      forward_op = [[
         if(input < -1)
            output = -1;
         else if(input <= 1)
            output = input;
         else
            output = 1
      ]],
      backward_op = [[
         if(input < -1 || input > 1)
            gradInput = 0;
         else
            gradInput = *gradOutput
      ]]
   },
}


for i,v in ipairs(definitions) do
   local forward_op = v.forward_op:gsub('output', 'x'):gsub('input', 'y')
   local backward_op = v.backward_op:gsub('gradInput', 'x'):gsub('gradOutput', 'z')
   
   torch.ClTensor.nn[v.name..'_updateOutput'] = function(self, input)
      self.output:resizeAs(input):apply2(input, forward_op)
      return self.output
   end
   
   torch.ClTensor.nn[v.name..'_updateGradInput'] = function(self, input, gradOutput)
      self.gradInput:resizeAs(input)
      if backward_op:find'output' then
         self.gradInput:apply3(self.output, gradOutput, (backward_op:gsub('output', 'y')))
      elseif backward_op:find'input' then
         self.gradInput:apply3(input, gradOutput, (backward_op:gsub('input', 'y')))
      end
      return self.gradInput
   end
end

