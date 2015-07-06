require 'nn'

-- monkey patch it in :-P

nn.SpatialConvolutionMM.base__init = nn.SpatialConvolutionMM.__init
nn.SpatialConvolutionMM.baseUpdateOutput = nn.SpatialConvolutionMM.updateOutput
nn.SpatialConvolutionMM.baseUpdateGradInput = nn.SpatialConvolutionMM.updateGradInput

function nn.SpatialConvolutionMM:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, groups)
  -- we're still going to use the normal constructor, just we will inject 'groups' into it
  print('creating spatialconvolutionmm, groups=', groups, 'inplane', nInputPlane, 'outplane', nOutputPlane)
  self:base__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  self.groups = groups
  print('created spatialconvolutionmm, groups=', groups)
end

function nn.SpatialConvolutionMM:updateOutput(input, target)
  if torch.type(input) ~= 'torch.ClTensor' then
    return self:baseUpdateOutput(input, target)
  end


  return self.output
end

function nn.SpatialConvolutionMM:updateGradInput(input, target)
  if torch.type(input) ~= 'torch.ClTensor' then
    return self:baseUpdateGradInput(input, target)
  end
  self.gradInput:resizeAs(input)
  self.gradInput:zero()


  return self.gradInput
end


