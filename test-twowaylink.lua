local ngh = require('nodeGraphHelper')

dict = {}
ngh.addTwoWayLink(dict, 'a')
ngh.addTwoWayLink(dict, 'b')
ngh.removeTwoWayLink(dict, 'a')
ngh.addTwoWayLink(dict, 'c')
for k, v in pairs(dict) do
  print(k, v)
end


