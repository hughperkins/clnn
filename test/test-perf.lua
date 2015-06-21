require 'cltorch'
require 'clnn'
require 'sys'

input = torch.Tensor(128, 32, 28, 28):uniform()

api = os.getenv('API')
if api == nil then
  api = 'cl'
end

layer = nn.Tanh()

if api == 'cl' then
  inputcl = input:cl()
  layercl = layer:cl()
  layercl:forward(inputcl)
  for i=1,200 do
    sys.tic()
    layercl:forward(inputcl)
    cltorch.finish()
    print('sys.toc()', sys.toc())
  end
end

