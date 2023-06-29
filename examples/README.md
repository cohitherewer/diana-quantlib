# Examples

Summary:

* `image_classification` on CIFAR10 dataset with several model architectures
* `anomaly_detection` on MNIST dataset with a Deep AutoEncoder (DAE)
* `keyword_spotting` on the Google Speech V0.02 dataset with DSCNN

Each example folder contains the following scripts:

* `train.py` for training a model on the dataset
* `quantize_ptq.py` for PTQ (Post-training quantization)
* `quantize_qat.py` for QAT (Quantization-aware training)

> **_NOTE:_** Below, we give an example for image classification, but these steps are the same for the other examples too.

## Setup

In addition to installing `dianaquantlib`, some examples require additional packages. To install these, run in this folder:

```
pip install -r requirements.txt
```

Create output directories for saving datasets, weights (.pth) and export files (.onnx)

```
cd image_classification
mkdir data checkpoints export
```

## Train a model (floating-point)

All examples automatically download their dataset if not already downloaded.

```
python train.py resnet8 checkpoints/resnet8.pth
```

## PTQ (8-bit)

```
python quantize_ptq.py resnet8 checkpoints/resnet8.pth -c config/resnet8.yaml
```

Then use `export/ResNet_QL_NOANNOTATION.onnx` as compilation input for TVM.

For estimating the power-of-two quantization parameters, the example uses the 'minmax'
estimator. However, it might be usefull to try 'meanstd' too and use whatever produces the
best accuracy.

## QAT (8-bit)

```
python quantize_qat.py mobilenetv1 checkpoints/mobilenetv1.pth -c config/mobilenetv1.yaml
```

This script will perform the following steps:

* Fake quantize model (FQ)
* Fine-tune ==> saves a `checkpoints/mobilenetv1.pth.fq` file with the highest accuracy
* Map to hardware (HW)
* Fine-tune ==> saves a `checkpoints/mobilenetv1.pth.hw` file with the highest accuracy
* Integerise and export to ONNX

If for example the accuracy of the FQ stage is satisfactory, but the accuracy of the HW stage is not,
one can rerun the HW stage again (after adjustments) with:

```
python quantize_qat.py mobilenetv1 checkpoints/mobilenetv1.pth.fq -c config/mobilenetv1.yaml
```

Or the export step with:

```
python quantize_qat.py mobilenetv1 checkpoints/mobilenetv1.pth.hw -c config/mobilenetv1.yaml
```

## Quantizing a custom model

### Define your model

Create a pytorch model in an `nn.Module` class as usual.
Preferrably avoid using `torch.nn.functional` as much as possible, use modules instead.
For example use `nn.ReLU` instead of `F.relu`.

### Train you model

Train you model in full precision as you would normally do in pytorch.

### Quantize you model

Create a script similar to `quantize_ptq.py` or `quantize_qat.py`.
In addition, create a `.yaml` config file. You can generate such a file with the help of `generate_config.py`. For example:

```
python generate_config.py resnet8 image_classification/config/resnet8.yaml
```

Look at the code on how to do this for your custom model.

> **_NOTE:_** By default, the file configures many layers for execution on to the analog core of DIANA. To map all layers to the digital core, find-replace `ANALOG` with `DIGITAL` in this file before using it for quantization.

