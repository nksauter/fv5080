EPOCH = OPT.epochNumber

local best_performance = 0
local train_performance = 0
local eval_performance = 0
local test_performance = 0

for i=OPT.epochNumber, OPT.nEpochs do
    if OPT.train then
        train_performance = train()
    end
    if OPT.eval and (i == OPT.nEpochs or i % OPT.nEpochsEval == 0) then
        eval_performance = eval()
    end
    if OPT.test and (i == OPT.nEpochs or i % OPT.nEpochsTest == 0) then
        test_performance = test()
    end

    -- Save the model
    if OPT.saveBest then
        -- assume larger the performance, better the performance
        if best_performance < eval_performance then
            best_performance = eval_performance
            saveDataParallel(paths.concat(OPT.save, 'best_model.t7'), MODEL)
        end
    end
    if (OPT.train
            and OPT.nEpochsSave > 0
            and (i == OPT.nEpochs or i % OPT.nEpochsSave == 0)) then
        saveDataParallel(paths.concat(OPT.save, 'model_' .. EPOCH .. '.t7'), MODEL)
    end
    EPOCH = EPOCH + 1
end
