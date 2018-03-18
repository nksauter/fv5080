#!/bin/bash
th main.lua \
-data "../" \
-cache checkpoint/multiple \
-dataset multiple_dataset \
-nDonkeys 8 \
-nEpochs 120 \
-epochSize 100 \
-batchSize 64 \
-iterSize 1 \
-netType simple_cnn \
-imageCrop 720 \
-kernelSize 5 \
-nEpochsSave 0 \
-saveBest \
-train -eval

