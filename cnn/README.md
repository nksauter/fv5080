This is the implementation of **A Convolutional Neural Network-Based Screening Tool for X-ray Serial Crystallography** by 
Tsung-Wei Ke, Aaron S. Brewster, Stella X. Yu, Daniela Ushizima, Chao Yang, and Nicholas Sauter (2018).

## Prerequisites
* Python2.7
* Torch7
* CUDA8
* CUDNN-8-v5

## Install Torch Modules
```
> luarocks install cunn
> luarocks install cudnn
> luarocks install hdf5
```

## Prepare `png` image data

### Download HDF5 Raw Data
1. Download raw data in HDF5 format from [here](http://cxidb.org/id-76.html)
2. Put the data in corresponding dir. For example, put 
   [lg36/r0087_2000.h5](http://cxidb.org/data/76/lg36/r0087_2000.h5) under ``data/LG36``

### Transform HDF5 Raw Data to PNG Image Data
```
> cd ../data
> python hdf5_to_png.py --set-name LO19 (/LN83/LN84/LG36/L498)
```

## Getting Started
Go back to ``cnn/`` dir, and get ready for training and testing.

### Train with single dataset
```
# Change SUBSET in scripts/train_single.sh to LO19/LN83/LN84/LG36/L498, and run the code.
> bash scripts/train_single.sh
```

The model will be saved as checkpoint/``SUBSET``/single_dataset/``simple_cnn,....``/``DATA_TIME``/best_model.t7

### Testing with single dataset
```
# Change SUBSET in scripts/test_single.sh to LO19/LN83/LN84/LG36/L498
# Change MODEL_PATH in scripts/test_single.sh to checkpoint/SUBSET/single_dataset/simple_cnn,..../DATA_TIME
# Run the code
> bash scripts/test_single.sh
```

### Train with multiple dataset
```
> bash scripts/train_multiple.sh
```

The model will be saved as checkpoint/multiple/multiple_dataset/``simple_cnn,....``/``DATA_TIME``/best_model.t7

### Testing with single dataset
```
# Change MODEL_PATH in scripts/test_multiple.sh to checkpoint/multiple/multiple_dataset/simple_cnn,..../DATA_TIME
# Run the code
> bash scripts/test_multiple.sh
```
