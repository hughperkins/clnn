-- from THCUNN.lua:

local ffi = require 'ffi'
local THNN = require 'nn.THNN'

local THCLNN = {}

-- load libTHCLNN
local libthclnn_searchpath = package.searchpath('libTHCLNN', package.cpath)
print('libthclnn_searchpath', libthclnn_searchpath)
THCLNN.C = ffi.load(libthclnn_searchpath)

local THCLNN_h = [[
typedef void THClState;

TH_API void THNN_ClAbs_updateOutput(
          THClState *state,
          THClTensor *input,
          THClTensor *output);
TH_API void THNN_ClAbs_updateGradInput(
          THClState *state,
          THClTensor *input,
          THClTensor *gradOutput,
          THClTensor *gradInput);
]]

local preprocessed = string.gsub(THCLNN_h, 'TH_API ', '')
ffi.cdef(preprocessed)

local THClState_ptr = ffi.typeof('THClState*')

function THCLNN.getState()
  return THClState_ptr(cltorch.getState());
end

local function extract_function_names(s)
  local t = {}
  for n in string.gmatch(s, 'TH_API void THNN_Cl([%a%d_]+)') do
    t[#t+1] = n
  end
  return t
end

-- build function table
local function_names = extract_function_names(THCLNN_h)

THNN.kernels['torch.ClTensor'] = THNN.bind(THCLNN.C, function_names, 'Cl', THCLNN.getState)
torch.getmetatable('torch.ClTensor').THNN = THNN.kernels['torch.ClTensor']

return THCLNN

