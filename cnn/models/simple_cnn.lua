local NET = {}
function NET.packages()
    require 'cudnn'
    cudnn.fastest = true
    cudnn.benchmark = true
    require 'utils/mathfuncs'
    require 'utils/utilfuncs'
end

function NET.createModel(opt)
    NET.packages()

    local Convolution = nn.SpatialConvolution
    local ReLU = nn.ReLU
    local SBatchNorm = nn.SpatialBatchNormalization

    local model = nn.Sequential()

    -- building block
    local function ConvBNReLU(...)
        local arg = {...}
        model:add(Convolution(unpack(arg)))
        model:add(SBatchNorm(arg[2]))
        model:add(ReLU(true))
        return model
    end

    local function Max(...)
        local arg = {...}
        model:add(nn.SpatialMaxPooling(unpack(arg)):ceil())
        return model
    end

    -- input 720*720
    ConvBNReLU(1, 4, 7, 7, 2, 2, 3, 3) -- 360 x 360
    Max(3, 3, 2, 2, 1, 1) -- 180 x 180
    ConvBNReLU(4, 8, 3, 3, 2, 2, 1, 1) -- 90 x 90
    Max(2, 2, 2, 2, 0, 0) -- 45 x 45
    ConvBNReLU(8, 16, 3, 3, 2, 2, 1, 1) -- 23 x 23
    Max(2, 2, 2, 2, 0, 0) -- 12 x 12
    ConvBNReLU(16, 32, 3, 3, 2, 2, 1, 1) -- 6 x 6
    Max(2, 2, 2, 2, 0, 0) -- 3 x 3

    local classifier = nn.Sequential()
    classifier:add(nn.View(-1, 32*3*3))
    classifier:add(nn.Linear(32*3*3, 3))
    classifier:add(nn.LogSoftMax())
    model:add(classifier)

    -- initialization from MSR
    local function MSRinit(net)
        local function init(name)
            for k,v in pairs(net:findModules(name)) do
                local n = v.kW*v.kH*v.nOutputPlane
                v.weight:normal(0,math.sqrt(2/n))
                v.bias:zero()
            end
        end
        -- have to do for both backends
        init('nn.SpatialConvolution')
        init('cudnn.SpatialConvolution')
    end

    MSRinit(model)

    return model
end

function NET.createCriterion()
    local criterion = nn.MultiCriterion()
    criterion:add(nn.ClassNLLCriterion())
    return criterion
end

function NET.trainOutputInit()
    local info = {}
    -- utilfuncs.newInfoEntry is defined in utils/train_eval_test_func.lua
    info[#info+1] = utilfuncs.newInfoEntry('loss',0,0)
    info[#info+1] = utilfuncs.newInfoEntry('top1',0,0)
    return info
end

function NET.trainOutput(info, outputs, labelsCPU, err, iterSize)
    local batch_size = outputs:size(1)
    local outputsCPU = outputs:float()
    assert(batch_size == labelsCPU:size(1))

    info[1].value   = err * iterSize
    info[1].N       = batch_size

    info[2].value   = mathfuncs.topK(outputsCPU, labelsCPU, 1)
    info[2].N       = batch_size
end

function NET.trainPerformance(info)
    return info[2].value / info[2].N
end

function NET.testOutputInit()
    local info = {}
    info[#info+1] = utilfuncs.newInfoEntry('loss',0,0)
    info[#info+1] = utilfuncs.newInfoEntry('top1',0,0)
    --info[#info+1] = utilfuncs.newInfoEntry('prediction',0,0,true)
    return info
end

function NET.testOutput(info, outputs, labelsCPU, err)
    local batch_size = outputs:size(1)
    local outputsCPU = outputs:float()
    info[1].value   = err * OPT.iterSize
    info[1].N       = batch_size

    info[2].value = mathfuncs.topK(outputsCPU, labelsCPU, 1)
    info[2].N     = batch_size

    --info[3].value = outputsCPU
end

function NET.testPerformance(info)
    return info[2].value / info[2].N
end

function NET.evalOutputInit()
    local info = {}
    info[#info+1] = utilfuncs.newInfoEntry('loss',0,0)
    info[#info+1] = utilfuncs.newInfoEntry('top1',0,0)
    return info
end

function NET.evalOutput(info, outputs, labelsCPU, err)
    local batch_size = outputs:size(1)
    local outputsCPU = outputs:float()
    info[1].value   = err * OPT.iterSize
    info[1].N       = batch_size

    info[2].value = mathfuncs.topK(outputsCPU, labelsCPU, 1)
    info[2].N     = batch_size
end

function NET.evalPerformance(info)
    return info[2].value / info[2].N
end

function NET.trainRule(currentEpoch)
    -- exponentially decay
    local delta = 3
    local start = 1 -- LR: 10^-(star) ~ 10^-(start + delta)
    local ExpectedTotalEpoch = OPT.nEpochs
    return {LR= 10^-((currentEpoch-1)*delta/(ExpectedTotalEpoch-1)+start),
            WD= 5e-4}
end

return NET
