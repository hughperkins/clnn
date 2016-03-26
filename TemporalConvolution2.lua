--[[

Compared to the base TemporalConvolution, this is:
- faster on GPUs (both in CUDA and in OpenCL)
- slower on CPUs
- the weights saved to file are incompatible with the original TemporalConvolution

Conceptually, it's a wrapper around the highly optimized SpatialConvolutionMM class

Just use it like TemporalConvolution, only with a '2' added to the class name.  Easy :-)

]]--

require 'torch'
require 'nn'

local TemporalConvolution2, parent = torch.class('nn.TemporalConvolution2', 'nn.Module')

function TemporalConvolution2:__init(inputFrameSize, outputFrameSize, kW, dW, padW)
   parent.__init(self)

  self.inputFrameSize = inputFrameSize
  self.outputFrameSize = outputFrameSize
  self.kW = kW
  self.dW = dW or 1
  self.padW = padW or 0
  self.sconv = nn.SpatialConvolution(inputFrameSize, outputFrameSize, 1, kW, 1, dW, 0, self.padW)
  self.weight = self.sconv.weight
  self.bias = self.sconv.bias
  self.gradWeight = self.sconv.gradWeight
  self.gradBias = self.sconv.gradBias
end

function TemporalConvolution2:clearState()
  self.sconv:clearState()
  parent:clearState()
end

function TemporalConvolution2:updateOutput(input)
  assert(input:dim() == 3, 'must provide batched input')
  local batchSize = input:size(1)
  local numFrames = input:size(2)
  local outFrames = numFrames - math.floor(self.kW/2)*2 + 2 * self.padW
  if self.kW%2 == 0 then outFrames = outFrames+1 end

  input = input:view(batchSize, numFrames, self.inputFrameSize, 1):transpose(2,3)
  local output = self.sconv:updateOutput(input):transpose(2,3)
  self.output:resize(batchSize, outFrames, self.outputFrameSize):copy(output)
  return self.output
end

function TemporalConvolution2:updateGradInput(input, gradOutput)
  assert(input:dim() == 3, 'must provide batched input')
  local batchSize = input:size(1)
  local numFrames = input:size(2)
  local outFrames = numFrames - math.floor(self.kW/2)*2 + 2 * self.padW
  if self.kW%2 == 0 then outFrames = outFrames+1 end

  input = input:view(batchSize, numFrames, self.inputFrameSize, 1):transpose(2,3)
  gradOutput = gradOutput:view(batchSize, outFrames, self.outputFrameSize, 1):transpose(2,3)
  local gradInput = self.sconv:updateGradInput(input, gradOutput):transpose(2,3)
  self.gradInput:resize(batchSize, numFrames, self.inputFrameSize):copy(gradInput)

  return self.gradInput
end

function TemporalConvolution2:accGradParameters(input, gradOutput, scale)
  assert(input:dim() == 3, 'must provide batched input')
  local batchSize = input:size(1)
  local numFrames = input:size(2)
  local outFrames = numFrames - math.floor(self.kW/2)*2 + 2 * self.padW
  if self.kW%2 == 0 then outFrames = outFrames+1 end

  input = input:view(batchSize, numFrames, self.inputFrameSize, 1):transpose(2,3)
  gradOutput = gradOutput:view(batchSize, outFrames, self.outputFrameSize, 1):transpose(2,3)
  self.sconv:accGradParameters(input, gradOutput, scale)  
end
