require 'nn'

-- monkey patch it in :-P

nn.SpatialConvolutionMM.base__init = nn.SpatialConvolutionMM.__init
nn.SpatialConvolutionMM.baseUpdateOutput = nn.SpatialConvolutionMM.updateOutput
nn.SpatialConvolutionMM.baseUpdateGradInput = nn.SpatialConvolutionMM.updateGradInput

-- k: filtersize (eg: 3, or 5)
-- d: stride size (eg: 1)
-- pad: padding
function nn.SpatialConvolutionMM:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, groups)
  -- we're still going to use the normal constructor, just we will inject 'groups' into it
--  print('creating spatialconvolutionmm, groups=', groups, 'inplane', nInputPlane, 'outplane', nOutputPlane)
  self:base__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  if groups == nil then
    return self
  end
  self.groups = groups
  self.groupInputPlane = nInputPlane / groups
  self.groupOutputPlane = nOutputPlane / groups
  self.calcedSize = false
  if self.groupInputPlane * groups ~= nInputPlane then
    error("groups must be a common factor of inputplane and outputplane")
  end
  if self.groupOutputPlane * groups ~= nOutputPlane then
    error("groups must be a common factor of inputplane and outputplane")
  end
  self.convs = {}
  for group=1,groups do
    self.convs[group] = self.new(self.groupInputPlane, self.groupOutputPlane, kW, kH, dW, dH, padW, padH)
  end
--  print('created spatialconvolutionmm, groups=', groups)
end

function nn.SpatialConvolutionMM:updateOutput(input)
  if torch.type(input) ~= 'torch.ClTensor' or self.groups == nil then
    return self:baseUpdateOutput(input, target)
  end
--  print('SpatialConvolutionMM:updateOutput')

  inDims = input:size():size()
  if inDims ~= 4 then
    error("SpatialConvolutionMM expects a 4d tensor currently")  -- keep it simple for now
  end
  if self.nInputPlane ~= input:size(2) then
    error("expecting " .. tostring( self.nInputPlane) .. ' input planes')
  end
--  print('input:size()', input:size())
  local bs = input:size(1)
  local inW = input:size(4)
  local inH = input:size(3)
  local outW = ( inW / self.dW + 2 * self.padW - self.kW ) + 1
  local outH = ( inH / self.dH + 2 * self.padH - self.kH ) + 1
--  print('sizes', bs, inH, inW, outH, outW)

  self.output:resize(torch.LongStorage({bs, self.nOutputPlane, outH, outW}))

--  print('self.groupInputPlane', self.groupInputPlane)
  for group=1,self.groups do
    inputslice = input:narrow(2, self.groupInputPlane * (group-1) + 1, self.groupInputPlane)
    if torch.type(self.convs[group]) ~= torch.type(input) then
      self.convs[group]:type(torch.type(input))
    end
--    print('types', torch.type(input), torch.type(self.convs[group]), torch.type(self.convs[group].weight))
--    print('child nInputPlane', self.convs[group].nInputPlane)
--    print('inputslice size', inputslice:size())
    self.convs[group].output = self.output:narrow(2, self.groupOutputPlane * (group-1) + 1, self.groupOutputPlane)
    self.convs[group]:updateOutput(inputslice)
  end

  return self.output
end

function nn.SpatialConvolutionMM:updateGradInput(input, target)
  if torch.type(input) ~= 'torch.ClTensor' or self.groups == nil then
    return self:baseUpdateGradInput(input, target)
  end
  self.gradInput:resizeAs(input)
  self.gradInput:zero()


  return self.gradInput
end


