require 'nngraph'
require 'clnn'

local x = nn.Identity()()
local n = nn.Linear(50,50)
local m = n(x)
local g = nn.gModule({x}, {m}):cl()
local input = torch.ClTensor(30,50):uniform()

local output = torch.ClTensor(30,30):uniform()
local addBuffer = torch.ClTensor(30):uniform()
local bias = torch.ClTensor(30):uniform()
local i = 1
while true do
  print('i', i)
  g:forward(input)
  --n:forward(input)
  --output:addr(1, addBuffer, bias)
  i = i + 1
end

