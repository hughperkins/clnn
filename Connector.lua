local Connector = torch.class('nn.Connector')

-- has two ends, which can each be attached to an endpoint,
-- or to another connector
function Connector:__init(names)
  assert(#names == 2)
  self.names = names
  self.namesSet = {}
  self.endpoints = {}
  for i, name in ipairs(names) do
    local endpoint = nn.Endpoint(self)
    self[name] = endpoint
    table.insert(self.endpoints, endpoint)
  end
  self.endpoints[1].connected = function() return self.endpoints[2] end
  self.endpoints[2].connected = function() return self.endpoints[1] end
end

--function Connector:join(name, second)
--  assert(self.namesSet[name])
--  assert(torch.type(second) == 'nn.Connector')
--  table.insert(self[name], second)
--end

---- give it one endpoint, it returns the other one
--function Connector:other(endpoint)
--  if self.endpoints[1] == endpoint then
--    return self.endpoints[2]
--  elseif self.endpoints[2] == endpoint then
--    return self.endpoints[1]
--  else
--    error("endpoint not in connector")
--  end
--end

