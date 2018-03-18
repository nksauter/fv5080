require 'image'
paths.dofile('dataset.lua')

local tf = require 'utils/transforms'

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(OPT.cache, 'trainCache.t7')
local evalCache = paths.concat(OPT.cache, 'evalCache.t7')
local testCache = paths.concat(OPT.cache, 'testCache.t7')
---------------------
-- Data Preprocess --
---------------------
local function loadImage(path)
    local img = image.load(path, 1, 'float')
    if img:dim() == 2 then img:resize(1, img:size(1), img:size(2)) end
    return img
end

local function lcn(size)
    local kernel = torch.DoubleTensor(size, size):fill(1/(size*size))
    return function (input)
        local output = image.lcn(input, kernel)
        if output:dim() == 2 then output:resize(1, output:size(1), output:size(2)) end
        return output
    end
end

local function reflect(size)
    local k = size
    return function (input)
        local c, h, w = unpack(input:size():totable())
        local rimg = input.new():resize(c, h+(k-1)*2, w+(k-1)*2):zero()
        rimg[{{}, {k,k+h-1}, {k,k+w-1}}] = input
            for i = 1,k-1 do
                rimg[{{}, {i}, {}}] = rimg[{{}, {2*k-i}, {}}]
                rimg[{{}, {i+h+(k-1)}, {}}] = rimg[{{}, {h+(k-1)-i}, {}}]
                rimg[{{}, {}, {i}}] = rimg[{{}, {}, {2*k-i}}]
                rimg[{{}, {}, {i+h+(k-1)}}] = rimg[{{}, {}, {h+(k-1)-i}}]
            end
        return rimg
    end
end

local function preprocess(img, split)
    local crop = split == 'train' and tf.RandomCrop or tf.CenterCrop
    local funcs = {
        crop(OPT.imageCrop),
        reflect(OPT.kernelSize),
        lcn(OPT.kernelSize),
        tf.CenterCrop(OPT.imageCrop)
    }
    return tf.Compose(funcs)(img)
end


--------------------------------------
-- DataSet split for train/val/test --
--------------------------------------
local split = {train=50, val=20, test=30}
---------------------------------
-- DataLoader for training set --
---------------------------------
local function trainHook(self, path)
    local img = loadImage(path)
    local proc_img = preprocess(img, 'train')
    collectgarbage()
    return proc_img
end

if paths.filep(trainCache) then
    print('Loading train metadata from cache')
    trainLoader = torch.load(trainCache)
    assert(trainLoader.path == paths.concat(OPT.data),
        'cached files dont have the same path as OPT.data. '
        .. 'Remove your cached files at: '
        .. trainCache .. ' and rerun the program')
else
    print('Creating train metadata')
    trainLoader = dataLoader{
        path = paths.concat(OPT.data),
        split = split,
        protocol = 'train',
    }
    torch.save(trainCache, trainLoader)
end
trainLoader.sampleHookTrain = trainHook

---------------------------------
-- DataLoader for val&test set --
---------------------------------
local function testHook(self,path)
    local img = loadImage(path)
    local proc_img = preprocess(img, 'test')
    collectgarbage()
    return proc_img
end

if paths.filep(evalCache) then
     print('Loading test metadata from cache')
     evalLoader = torch.load(evalCache)
     assert(evalLoader.path == paths.concat(OPT.data),
         'cached files dont have the same path as OPT.data. '
         .. 'Remove your cached files at: '
         .. evalCache .. ' and rerun the program')
else
    print('Creating val metadata')
    evalLoader = dataLoader{
        path = paths.concat(OPT.data),
        split = split,
        protocol = 'val',
    }
    torch.save(evalCache, evalLoader)
end
evalLoader.sampleHookTest = testHook

if paths.filep(testCache) then
     print('Loading test metadata from cache')
     testLoader = torch.load(testCache)
     assert(testLoader.path == paths.concat(OPT.data),
         'cached files dont have the same path as OPT.data. '
         .. 'Remove your cached files at: '
         .. testCache .. ' and rerun the program')
else
    print('Creating test metadata')
    testLoader = dataLoader{
        path = paths.concat(OPT.data),
        split = split,
        protocol = 'test',
    }
    torch.save(testCache, testLoader)
end
testLoader.sampleHookTest = testHook
