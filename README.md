# CVPR 2023 paper - Hard Sample Matters a Lot in Zero-Shot Quantization [paper]()

## Requirements

Python >= 3.7.10

Pytorch == 1.8.1

## Reproduce results

### Stage1: Generate data.

take cifar10 as an example:
```
cd data_generate
```
"--save_path_head" in **run_generate_cifar10.sh/run_generate_imagenet.sh** is the path where you want to save your generated data pickle.

```
bash run_generate_cifar10.sh
```


### Stage2: Train the quantized network

```
cd ..
```
1. Modify "qw" and "qa" in cifar10_resnet20.hocon to select desired bit-width.

2. Modify "dataPath" in cifar10_resnet20.hocon to the real dataset path (for construct the test dataloader).

3. Modify "generateDataPath" and ""generateLabelPath" in cifar10_resnet20.hocon to the data_path and label_path you just generate from Stage1.

4. Use the commands in run.sh to train the quantized network. Please note that the model that generates the data and the quantized model should be the same.
