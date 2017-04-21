require 'nn'

nn.SpatialFullConvolution.baseUpdateOutput = nn.SpatialFullConvolution.updateOutput
nn.SpatialFullConvolution.baseUpdateGradInput = nn.SpatialFullConvolution.updateGradInput
nn.SpatialFullConvolution.baseAccGradParameters = nn.SpatialFullConvolution.accGradParameters

function nn.SpatialFullConvolution:updateOutput(input)
  if torch.type(input) ~= 'torch.ClTensor' then
    return self:baseUpdateOutput(input, target)
  end
  self:backCompatibility()

  local inputTensor = input
  local adjW, adjH = self.adjW, self.adjH

  -- The input can be a table where the second element indicates the target
  -- output size, in which case the adj factors are computed automatically
  if type(inputTensor) == 'table' then
    inputTensor = input[1]
    local targetTensor = input[2]
    local tDims = targetTensor:dim()
    local tH = targetTensor:size(tDims-1)
    local tW = targetTensor:size(tDims)
    adjW = calculateAdj(tW, self.kW, self.padW, self.dW)
    adjH = calculateAdj(tH, self.kH, self.padH, self.dH)
    self.finput = self.finput or input[1].new()
    self.fgradInput = self.fgradInput or input[1].new()
  else
    self.finput = self.finput or input.new()
    self.fgradInput = self.fgradInput or input.new()
  end

  inputTensor = makeContiguous(self, inputTensor)
  inputTensor.THNN.SpatialFullConvolution_updateOutput(
    inputTensor:cdata(),
    self.output:cdata(),
    self.weight:cdata(),
    THNN.optionalTensor(self.bias),
    self.finput:cdata(),
    self.fgradInput:cdata(),
    self.nInputPlane,
    self.nOutputPlane,
    self.kW, self.kH,
    self.dW, self.dH,
    self.padW, self.padH,
    adjW, adjH
  )

  return self.output
end

function nn.SpatialFullConvolution:updateGradInput(input, gradOutput)
  if torch.type(input) ~= 'torch.ClTensor' then
    return self:baseUpdateGradInput(input, gradOutput)
  end
  self:backCompatibility()

  if self.gradInput then

    local inputTensor = input
    local adjW, adjH = self.adjW, self.adjH

    -- The input can be a table where the second element indicates the target
    -- output size, in which case the adj factors are computed automatically
    if type(inputTensor) == 'table' then
      inputTensor = input[1]
      local targetTensor = input[2]
      local tDims = targetTensor:dim()
      local tH = targetTensor:size(tDims-1)
      local tW = targetTensor:size(tDims)
      adjW = calculateAdj(tW, self.kW, self.padW, self.dW)
      adjH = calculateAdj(tH, self.kH, self.padH, self.dH)
      -- Momentarily extract the gradInput tensor
      if type(self.gradInput) == 'table' then
        self.gradInput = self.gradInput[1] or inputTensor.new()
      end
    end

    inputTensor, gradOutput = makeContiguous(self, inputTensor, gradOutput)
    inputTensor.THNN.SpatialFullConvolution_updateGradInput(
      inputTensor:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.weight:cdata(),
      self.finput:cdata(),
      self.nInputPlane,
      self.nOutputPlane,
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      adjW, adjH
    )

    if type(input) == 'table' then
     -- Create a zero tensor to be expanded and used as gradInput[2].
      self.zeroScalar = self.zeroScalar or input[2].new(1):zero()
      self.ones:resize(input[2]:dim()):fill(1)
      local zeroTensor =  self.zeroScalar
          :view(table.unpack(self.ones:totable()))
          :expandAs(input[2])
      self.gradInput = {self.gradInput, zeroTensor}
    end

    return self.gradInput
  end
end

function nn.SpatialFullConvolution:accGradParameters(input, gradOutput, scale)
  if torch.type(input) ~= 'torch.ClTensor' then
    return self:baseAccGradParameters(input, gradOutput)
  end
  scale = scale or 1
  self:backCompatibility()

  local inputTensor = input
  local adjW, adjH = self.adjW, self.adjH

  -- The input can be a table where the second element indicates the target
  -- output size, in which case the adj factors are computed automatically
  if type(inputTensor) == 'table' then
    inputTensor = input[1]
    local targetTensor = input[2]
    local tDims = targetTensor:dim()
    local tH = targetTensor:size(tDims-1)
    local tW = targetTensor:size(tDims)
    adjW = calculateAdj(tW, self.kW, self.padW, self.dW)
    adjH = calculateAdj(tH, self.kH, self.padH, self.dH)
  end

  inputTensor, gradOutput = makeContiguous(self, inputTensor, gradOutput)
  inputTensor.THNN.SpatialFullConvolution_accGradParameters(
    inputTensor:cdata(),
    gradOutput:cdata(),
    self.gradWeight:cdata(),
    THNN.optionalTensor(self.gradBias),
    self.finput:cdata(),
    self.fgradInput:cdata(),
    self.finput:cdata(),
    self.fgradInput:cdata(),
    self.kW, self.kH,
    self.dW, self.dH,
    self.padW, self.padH,
    adjW, adjH,
    scale
  )
end
