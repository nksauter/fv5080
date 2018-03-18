#!/bin/bash
MODEL_PATH=""

th main.lua \
-data "../" \ 
-cache checkpoint/multiple \
-dataset multiple_dataset \
-nDonkeys 8 \
-nEpochs 1 \
-epochSize 1 \
-batchSize 64 \
-iterSize 1 \
-netType simple_cnn \
-imageCrop 720 \
-kernelSize 5 \
-nEpochsSave 0 \
-retrain $MODEL_PATH/best_model.t7 \
-saveBest \
-test
