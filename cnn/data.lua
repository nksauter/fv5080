local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
do -- start K datathreads (donkeys)
   if OPT.nDonkeys > 0 then
      local options = OPT -- make an upvalue to serialize over to donkey threads
      DONKEYS = Threads(
         OPT.nDonkeys,
         function()
            require 'torch'
         end,
         function(idx)
            OPT = options -- pass to all donkeys via upvalue
            tid = idx
            local seed = OPT.manualSeed + idx
            torch.manualSeed(seed)
            print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            paths.dofile('dataset/' .. OPT.dataset .. '/donkey.lua')
         end
      );
   else -- single threaded data loading. useful for debugging
      paths.dofile('dataset/' .. OPT.dataset .. '/donkey.lua')
      DONKEYS = {}
      function DONKEYS:addjob(f1, f2) f2(f1()) end
      function DONKEYS:synchronize() end
   end
end

if OPT.eval then
   NEVAL = 0
   DONKEYS:addjob(function() return evalLoader:size() end, function(c) NEVAL = c end)
   DONKEYS:synchronize()
   assert(NEVAL > 0, "Failed to get NEVAL")
   print('NEVAL: ', NEVAL)
end

if OPT.test then
   NTEST = 0
   DONKEYS:addjob(function() return testLoader:size() end, function(c) NTEST = c end)
   DONKEYS:synchronize()
   assert(NTEST > 0, "Failed to get NTEST")
   print('NTEST: ', NTEST)
end
