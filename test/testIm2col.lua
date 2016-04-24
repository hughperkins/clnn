require 'clnn'

local THNN = require 'nn.THNN'

local ok, ffi = pcall(require, 'ffi')

require 'torch'

--typedef void THClState;

--typedef struct THClTensor
--{
--    long *size;
--    long *stride;
--    int nDimension;
--   
--} THClTensor;


ffi.cdef[[
typedef void THClState;

void im2col(THClState *state, THClTensor* im,
    const int nInputPlane, 
    const int inW, const int inH,
    const int kW, const int kH,
    const int dW, const int dH,
    const int padW, const int padH,
    THClTensor* col);

void im2col_batched(THClState *state, THClTensor* im,
    const int nInputPlane, 
    const int inW, const int inH,
    const int kW, const int kH,
    const int dW, const int dH,
    const int padW, const int padH,
    const int numImages, const int imageIdx,
    THClTensor* col);
]]

local THClState_ptr = ffi.typeof('THClState*')

local function getState()
  return THClState_ptr(cltorch.getState());
end

local libthclnn_searchpath = package.searchpath('libTHCLNN', package.cpath)
--print('libthclnn_searchpath', libthclnn_searchpath)
local thclnn = ffi.load(libthclnn_searchpath)

print('thclnn', thclnn)

print('thclnn.im2col', thclnn.im2col)

require 'cltorch'
--a = torch.ClTensor(3,3):uniform()
--b = torch.ClTensor(9,9):uniform()
--thclnn.im2col(getState(), a:cdata(), 3, 1, 1, 1, 1, 1, 1, 1, 1, b:cdata())

-- input im is [inPlanes][inH][inW]
-- output col is [inPlanes * kH * kW][outH * outW]
function manualIm2col(im, inPlanes, inW, inH, kW, kH, dW, dH, padH, padW)
  local outW = (inW + 2*padW - kW) / dW + 1;
  local outH = (inH + 2*padH - kH) / dH + 1;
  local col = torch.FloatTensor(inPlanes * kH * kW, outH * outW):zero()

  for outh=1,outH do
    for outw=1,outW do
--      local inGrid = torch.FloatTensor(kH, kW):zero()
        
--      local 
      for inPlane=1,inPlanes do
        for kh = 1,kH do
          local inh = outh + kh - 1 - padH
          for kw=1,kW do
            local inw = outw + kw - 1 - padW
            if inh >= 1 and inh <= inH and inw >= 1 and inw <= inW then
              local invalue = im[inPlane][inh][inw]
--              print('outh=' .. outh .. ' outw=' .. outw .. ' inplane=' .. inPlane .. ' kh= ' .. kh .. ' kw=' .. kw .. ' invalue=' .. invalue)
--              print('row', inPlane * kH * kW + kh * kW + kw)
--              print('col', outh * outW + outw)
--              print('col:size()', col:size())
              col[(inPlane-1) * kH * kW + (kh-1) * kW + kw][(outh-1) * outW + outw] = invalue
            end
          end
    --    for h_out=1,outH do
        end
      end
    end
  end
  return col
end

function manualIm2col_batched(images, inPlanes, inW, inH, kW, kH, dW, dH, padH, padW, numImages)
  local outW = (inW + 2*padW - kW) / dW + 1;
  local outH = (inH + 2*padH - kH) / dH + 1;
  local col = torch.FloatTensor(inPlanes * kH * kW, numImages * outH * outW):zero()

  for i=1,numImages do
    local im = images[i]
--    print('im:size()', im:size())
    for outh=1,outH do
      for outw=1,outW do
  --      local inGrid = torch.FloatTensor(kH, kW):zero()
          
  --      local 
        for inPlane=1,inPlanes do
          for kh = 1,kH do
            local inh = outh + kh - 1 - padH
            for kw=1,kW do
              local inw = outw + kw - 1 - padW
              if inh >= 1 and inh <= inH and inw >= 1 and inw <= inW then
                local invalue = im[inPlane][inh][inw]
  --              print('outh=' .. outh .. ' outw=' .. outw .. ' inplane=' .. inPlane .. ' kh= ' .. kh .. ' kw=' .. kw .. ' invalue=' .. invalue)
  --              print('row', inPlane * kH * kW + kh * kW + kw)
  --              print('col', outh * outW + outw)
  --              print('col:size()', col:size())
                col[(inPlane-1) * kH * kW + (kh-1) * kW + kw][(i-1) * outW * outH + (outh-1) * outW + outw] = invalue
              end
            end
      --    for h_out=1,outH do
          end
        end
      end
    end
  end
  return col
end


local inPlanes = 1
local outPlanes = 1
local inW = 3
local inH = 3
local kW = 3
local kH = 3
local dW = 1
local dH = 1
local padW = 1
local padH = 1

local outW = inW - 2 * math.floor(kW/2) + 2 * padW 
local outH = inH - 2 * math.floor(kH/2) + 2 * padH
print('outW', outW, 'outH', outH)

local im = torch.ClTensor({{2,5,4},
                     {9,7,3},
                     {6,8,1}})
im = im:reshape(inPlanes, inH, inW)
print('im', im)
local col = torch.ClTensor(inPlanes * kH * kW, outH * outW):zero()
thclnn.im2col(getState(), im:cdata(), inPlanes, inW, inH, kW, kH, dW, dH, padW, padH, col:cdata())
print('col', col)

local colman = manualIm2col(im, inPlanes, inW, inH, kW, kH, dW, dH, padW, padH)
print('colman', colman)

assert((colman:float() - col:float()):abs():max() < 0.0001)


local inPlanes = 2
local inW = 3
local inH = 3
local kW = 3
local kH = 3
local dW = 1
local dH = 1
local padW = 1
local padH = 1

local outW = inW - 2 * math.floor(kW/2) + 2 * padW 
local outH = inH - 2 * math.floor(kH/2) + 2 * padH
print('outW', outW, 'outH', outH)

local im = torch.ClTensor({
   {{2,5,4},
   {9,7,3},
   {6,8,1}},

   {{7,5,3},
   {2,4,6},
   {9,5,2}},
})
--im = im:reshape(inPlanes, inH, inW)
print('im', im)
local col = torch.ClTensor(inPlanes * kH * kW, outH * outW):zero()
thclnn.im2col(getState(), im:cdata(), inPlanes, inW, inH, kW, kH, dW, dH, padW, padH, col:cdata())
print('col', col)

local colman = manualIm2col(im, inPlanes, inW, inH, kW, kH, dW, dH, padW, padH)
print('colman', colman)

assert((colman:float() - col:float()):abs():max() < 0.0001)



local numImages = 2
local inPlanes = 1
local inW = 3
local inH = 3
local kW = 3
local kH = 3
local dW = 1
local dH = 1
local padW = 0
local padH = 0

local outW = inW - 2 * math.floor(kW/2) + 2 * padW 
local outH = inH - 2 * math.floor(kH/2) + 2 * padH
print('outW', outW, 'outH', outH)

local im = torch.ClTensor({
   {{2,5,4},
   {9,7,3},
   {6,8,1}},

   {{7,5,3},
   {2,4,6},
   {9,5,2}},
})
im = im:reshape(numImages, inPlanes, inH, inW)
print('im', im)
local col = torch.ClTensor(inPlanes * kH * kW, numImages * outH * outW):zero()
--print('col', col)
thclnn.im2col_batched(getState(), im[1]:cdata(), inPlanes, inW, inH, kW, kH, dW, dH, padW, padH, numImages, 0, col:cdata())
--print('col after im2col batched 1', col)
thclnn.im2col_batched(getState(), im[2]:cdata(), inPlanes, inW, inH, kW, kH, dW, dH, padW, padH, numImages, 1, col:cdata())
print('col after im2col batched 2', col)

local colman = manualIm2col_batched(im, inPlanes, inW, inH, kW, kH, dW, dH, padW, padH, numImages)
print('colman', colman)

assert((colman:float() - col:float()):abs():max() < 0.0001)




local numImages = 2
local inPlanes = 1
local inW = 3
local inH = 3
local kW = 3
local kH = 3
local dW = 1
local dH = 1
local padW = 1
local padH = 1

local outW = inW - 2 * math.floor(kW/2) + 2 * padW 
local outH = inH - 2 * math.floor(kH/2) + 2 * padH
print('outW', outW, 'outH', outH)

local im = torch.ClTensor({
   {{2,5,4},
   {9,7,3},
   {6,8,1}},

   {{7,5,3},
   {2,4,6},
   {9,5,2}},
})
im = im:reshape(numImages, inPlanes, inH, inW)
print('im', im)
local col = torch.ClTensor(inPlanes * kH * kW, numImages * outH * outW):zero()
--print('col', col)
thclnn.im2col_batched(getState(), im[1]:cdata(), inPlanes, inW, inH, kW, kH, dW, dH, padW, padH, numImages, 0, col:cdata())
--print('col', col)
thclnn.im2col_batched(getState(), im[2]:cdata(), inPlanes, inW, inH, kW, kH, dW, dH, padW, padH, numImages, 1, col:cdata())
print('col', col)

local colman = manualIm2col_batched(im, inPlanes, inW, inH, kW, kH, dW, dH, padW, padH, numImages)
print('colman', colman)

assert((colman:float() - col:float()):abs():max() < 0.0001)


--// columns is [inPlanes * kH * kW][outH * outW]
--// weight is [outplanes][inplanes * kH * kW]
--// output is [outPlanes][outH][outW]

--TH_API void im2col(THClState *state, THClTensor* im,
--    const int nInputPlane, 
--    const int inW, const int inH,
--    const int kW, const int kH,
--    const int dW, const int dH,
--    const int padW, const int padH,
--    THClTensor* col);

