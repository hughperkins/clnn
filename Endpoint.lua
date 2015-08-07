local Endpoint = torch.class('nn.Endpoint')

function Endpoint:__init(fixture)
  self.fixture = fixture
  self.attached = {}
end

function Endpoint:attach(otherEndpoint)
  assert(torch.type(otherEndpoint) == 'nn.Endpoint')
  assert(not self.attached[otherEndpoint])
  assert(not otherEndpoint.attached[self])
  self.attached[otherEndpoint] = true
  otherEndpoint.attached[self] = true
end

function Endpoint:detach(otherEndpoint)
  assert(torch.type(otherEndpoint) == 'nn.Endpoint')
  assert(self.attached[otherEndpoint])
  assert(otherEndpoint.attached[self])
  self.attached[otherEndpoint] = nil
  otherEndpoint.attached[self] = nil
end

function Endpoint:numAttached()
  local count = 0
  for _, _ in pairs(self.attached) do
    count = count + 1
  end
  return count
end

function Endpoint:other()
  local count = 0
  local other = nil
  for another, _ in pairs(self.attached) do
    other = another
    count = count + 1
  end
  assert(count <= 1)
  return other
end

