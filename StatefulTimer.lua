local StatefulTimer = torch.class('nn.StatefulTimer')

function StatefulTimer:__init(prefix)
  self.times = {}
  self.counts = {}
  if prefix == nil then
    prefix = ''
  end
  self.prefix = prefix
  self.timer = torch.Timer()
  self.last = self.timer:time().real
end

function StatefulTimer:state(name)
  if self.times[name] == nil then
    self.times[name] = 0
    self.counts[name] = 0
  end
  now = self.timer:time().real
  change = now - self.last
  self.last = now
  self.times[name] = self.times[name] + change
  self.counts[name] = self.counts[name] + 1
end

function StatefulTimer:dump()
  for k,v in pairs(self.times) do
    print(self.prefix .. k, v, self.counts[k])
  end
end

