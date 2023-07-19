# diana-training-fmw
Quantization-aware training framework for DIANA

# Install

Clone this repo:

```
git clone https://github.com/dianaKUL/diana-training-fmw.git
cd diana-training-fmw
git submodule init
git submodule update
```

With python virtual env and virtualenvwrapper:

```
mkvirtualenv quantlib -p /usr/bin/python3.10
workon quantlib
pip install -e .
```

# Run examples

See the [examples](examples/README.md)

# Diana quantlib API

The following steps are executed to quantize a pytorch model:

```python
from dianaquantlib.utils.BaseModules import DianaModule

model = MyModel()

def representative_dataset():
    for _, data in dataloader:
        yield data

# create a fake quantized model from a regular pytorch model
fq_model = DianaModule(
    DianaModule.from_trainedfp_model(model),
    representative_dataset,
)

# enable weights and activation quantization (fake quantized)
fq_model.set_quantized(activations=True)

# rewrite operations to match the hardware (still fake quantized)
fq_model.map_to_hw()

# convert all decimal values to integer values (true quantized)
fq_model.integrize_layers()

# export to onnx
fq_model.export_model('export')
```

1. `DianaModule` prepares the model for quantization:
    * Apply canonicalization: replace functionals such as `F.relu` with `nn.ReLU`
    * Wraps modules like `nn.Conv2d` and `nn.ReLU` in `DIANAConv2d` and `DIANAReLU` modules, respectively

2. `fq_model.set_quantized` enables the quantization wrappers:
    * The fake quantized model is forwarded with samples from the `representative_dataset` function. During forward, layer-wise statistics are recorded
    * Based on these statistics, quantization scales are estimated for each layer
    * Quantization is enabled
    * After this step, fine-tuning can be done if desired, since quantization can introduce loss in accuracy

3. `fq_model.map_to_hw`:
    * Batchnorm layers are folded into their preceding convolution layers, in case they are mapped to the digital core
    * Quantization scales are re-calculated with the same procedure as the previous step
    * `DIANAReLU` and `DIANAIdentity` modules are replaced with re-quantizer modules. This is part of the integrize step, not sure why this is done in this step
    * After this step, additional fine-tuning can be done if desired, since folding batchnorm layers plus re-calculating the scales can introduce loss in accuracy

4. `fq_model.integrize_layers`:
    * Rescale weights and biases to real integers

5. `fq_model.export_model`:
    * Export quantized model to ONNX file (.onnx file)
    * Dump test input/intermediate/output tensor maps (.npy files)


# Supported models and operations

Tested models:

* ResNet8
* ResNet20
* MobileNetV1
* MobileNetV2
* DeepAutoEncoder (DAE)
* DSCNN

## Supported PyTorch modules

* `nn.AdaptiveAvgPool2d`
* `nn.BatchNorm2d`
* `nn.Conv2d`
* `nn.Dropout`
* `nn.Dropout2d`
* `nn.Flatten`
* `nn.Linear`
* `nn.ReLU`

> **_NOTE:_** Although all convolution hyper parameters are supported, the accelerator cores on DIANA only support a limited set. See HTVM's documentation for more info.

## Supported functions and methods

* `add`
* `flatten`
* `F.relu`
* `reshape`
* `view`

## Supported quantization schemes

### Digital core

* symmetric per-axis int8 weight quantization with power-of-two scale
* symmetric per-axis int8 activation quantization with power-of-two scale

### Analog core

TODO

> **_WARNING:_** Support for the analog core is unfinished

### CPU

Currently, dianaquantlib assumes a quantized op is ran on either the digital or analog core. It therefore either uses the quantization scheme of the digital or analog core.
In the future, more flexible quantization schemes could be used when ran on CPU.
