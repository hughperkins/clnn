require 'nn'
require 'clnn'

local bs = 8
local from = 32
local to = 32
local ki = 5
local kj = 5
local si = 1
local sj = si
local padi = 2
local padj = 2
local outi = 28
local outj = 28
local ini = (outi-1)*si+ki
local inj = (outj-1)*sj+kj

local groups = 4

--local tm = {}
--local title = string.format('SpatialConvolutionMM.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]',
--                           bs, from, inj, ini, kj, ki, bs, to, outj, outi, sj, si)
--times[title] = tm

local input = torch.randn(bs,from,inj,ini)
local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj)
--local groundtruth = sconv:forward(input)
--local a = torch.Timer()
--for i = 1,nloop do
--  groundtruth = sconv:forward(input)
--end
--tm.cpu = a:time().real

input = input:cl()
local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padi,padj,groups):cl()
gconv.weight = sconv.weight:cl()
gconv.bias = sconv.bias:cl()
local rescl = gconv:forward(input)
--a:reset()
--for i = 1,nloop do
  rescl = gconv:forward(input)
--end
--cltorch.synchronize()
--tm.gpu = a:time().real

--local error = rescl:float() - groundtruth
-- mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')


