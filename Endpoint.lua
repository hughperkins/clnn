local Endpoint = torch.class('nn.Endpoint')

-- Endpoint is attached permanently to a single fixture, and can be connected to one or more other Connectors
function Endpoint:__init(fixture)
  self.fixture = fixture
  self._attached = {}
end

function Endpoint:attach(connector)
  assert(torch.type(connector) == 'nn.Endpoint')
  assert(not self._attached[connector])
  assert(not connector._attached[self])
  self._attached[connector] = true
  connector._attached[self] = true
end

function Endpoint:detach(connector)
  assert(torch.type(connector) == 'nn.Endpoint')
  assert(self._attached[connector])
  assert(connector._attached[self])
  self._attached[connector] = nil
  connector._attached[self] = nil
end

function Endpoint:numAttached()
  local count = 0
  for _, _ in pairs(self._attached) do
    count = count + 1
  end
  return count
end

function Endpoint:attached()
  local count = 0
  local other = nil
  for another, _ in pairs(self._attached) do
    other = another
    count = count + 1
  end
  assert(count <= 1)
  return other
end

