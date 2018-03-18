require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local argcheck = require 'argcheck'

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
    pack=true,
    help=[[
        A dataset class for dataset in a folder with images/ and img_list.txt
        and annotation.txt
    ]],
    {name="path",
     type="string",
     help="Path of directory to the dataset"},

    {name="split",
     type="table",
     help="Percentage of split to go to Training"
    },

    {name="protocol",
     type="string",
     help="train | test"},
}

local function readAnno(txtPath)
    local file = io.open(txtPath, 'r')
    local str = file:read("*all")
    file:close()
    local lines = str:split("\n")

    local annos = {}
    for _, line in ipairs(lines) do
        local dset, id, label = unpack(line:split('%s+'))
        id = id:gsub('%[', ''):gsub('%]', '')
        id = string.format('%05d', tonumber(id))

        if label == 'MISS' then
            annos[id] = 3
        elseif label == 'HIT' then
            annos[id] = 1
        elseif label == 'MAYBE' then
            annos[id] = 2
        else
            error('Unknow annotation!')
        end
    end
    return annos
end

local function readImage(txtPath)
    local file = io.open(txtPath, 'r')
    local str = file:read("*all")
    file:close()
    return str:split("\n")
end

local function tb2CharTensor(tb, rootDir)
    assert(#tb > 1)
    local maxStr = -1
    for i,line in ipairs(tb) do
        tb[i] = paths.concat(rootDir, line)
        maxStr = maxStr > #tb[i] and maxStr or #tb[i]
    end

    maxStr = maxStr + 1
    local tensor = torch.CharTensor(#tb, maxStr)
    local s_data = tensor:data()
    for i,line in ipairs(tb) do
        ffi.copy(s_data, line)
        s_data = s_data + maxStr
    end
    return tensor
end

function dataset:__init(...)
    local args =  initcheck(...)
    for k,v in pairs(args) do self[k] = v end

    local imagePaths, imageClasses = {}, {}
    for _, set in ipairs{'LO19', 'LN83', 'LN84', 'LG36', 'L498'} do
        local imgPathList = paths.concat(
            self.path, 'data', set, 'images.txt'
        )
        local imgPath = readImage(imgPathList)
        local imgClassList = paths.concat(
            self.path, 'annotation', set,
            'expert_annotation.txt'
        )
        local imgClassById = readAnno(imgClassList)
        local imgClass = {}
        for i,line in ipairs(imgPath) do
            local tmp = line:split('/')
            local id = tmp[#tmp]:gsub('.png', '')
            local class = imgClassById[id] or 2 -- if not annotated, there's no bragg spot
            imgClass[i] = class
        end

        local pathTensor = tb2CharTensor(imgPath, self.path)
        local classTensor = torch.FloatTensor(imgClass)
        -- split for train/val
        local nData = pathTensor:size(1)
        local startInd, nSplit
        if self.protocol == 'train' then
            startInd = 1
            nSplit = math.floor(nData*self.split.train/100)
        elseif self.protocol == 'val' then
            startInd = math.floor(nData*self.split.train/100)+1
            nSplit = math.ceil(nData*self.split.val/100)
        else
            startInd = math.ceil(nData*(self.split.train+self.split.val)/100)+1
            nSplit = math.floor(nData*self.split.test/100)
        end
        nSplit = nSplit == 0 and 1 or nSplit

        imagePaths[#imagePaths+1] = pathTensor:narrow(1, startInd, nSplit)
        imageClasses[#imageClasses+1] = classTensor:narrow(1, startInd, nSplit)
    end

    self.imagePath = torch.cat(imagePaths, 1)
    self.imageClass = torch.cat(imageClasses, 1)

    -- cached information
    self.classes = {'HIT', 'MAYBE', 'MISS'}
    self.classListSample = {}
    for i = 1,self.imageClass:size(1) do
        local class = self.imageClass[i]
        self.classListSample[class] = self.classListSample[class] or {}
        table.insert(self.classListSample[class], i)
    end
end

-- size()
function dataset:size()
    return self.imagePath:size(1)
end

-- getByClass
function dataset:getByClass(class)
    local index = math.max(
        1,
        math.ceil(torch.uniform() * #self.classListSample[class])
    )
    local imgpath = ffi.string(
        torch.data(self.imagePath[self.classListSample[class][index]])
    )
    return self:sampleHookTrain(imgpath)
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, tab)
    local tensor
    local quantity = #tab
    local iSize = torch.isTensor(tab[1]) and tab[1]:size():totable() or {}
    local tSize = {quantity}
    for _, dim in ipairs(iSize) do table.insert(tSize, dim) end
    tensor = torch.Tensor(table.unpack(tSize)):fill(-1)
    for i=1,quantity do
        tensor[i] = tab[i]
    end
    return tensor
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
    assert(quantity)
    local dataTable = {}
    local scalarTable = {}
    for i=1,quantity do
        local class = torch.random(1, #self.classes)
        local out = self:getByClass(class)
        table.insert(dataTable, out)
        table.insert(scalarTable, class)
    end
    local data = tableToOutput(self, dataTable)
    local scalarLabels = tableToOutput(self, scalarTable)
    return data, scalarLabels
end

function dataset:genInputs(quantity, currentEpoch)
    local data, scalarLabels = self:sample(quantity)
    return {data}, {scalarLabels}
end

function dataset:get(i1, i2)
    local indices = torch.range(i1, i2);
    local quantity = i2 - i1 + 1;
    assert(quantity > 0)
    -- now that indices has been initialized, get the samples
    local dataTable = {}
    local scalarTable = {}
    for i=1,quantity do
        -- load the sample
        local ind = indices[i]
        local imgpath = ffi.string(torch.data(self.imagePath[ind]))
        local out = self:sampleHookTest(imgpath)
        table.insert(dataTable, out)
        table.insert(scalarTable, self.imageClass[ind])
    end
    local data = tableToOutput(self, dataTable)
    local scalarLabels = tableToOutput(self, scalarTable)
    return data, scalarLabels
end

function dataset:getInputs(i1, i2, currentEpoch)
    local data, scalarLabels = self:get(i1, i2)
    return {data}, {scalarLabels}
end

return dataset
