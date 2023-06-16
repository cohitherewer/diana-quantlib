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

With python virtual env:

```
mkvirtualenv quantlib -p /usr/bin/python3.10
workon quantlib
pip install -e .
```

# Run examples

## Post-training quantization

First train the model in full-precision:

```
cd examples/cifar10
mkdir checkpoints export
python train.py
```

Then apply PTQ:

```
python quantize_ptq.py
```

# Workflow
- ## Pytorch Model 
  The first step is to define a pytorch nn module and making sure that it is functional. If possible,  always use pytorch **nn.Module**'s as the building blocks for you model to ensure compatibility with the library. Non-modular functions should be avoided when possible. For example, it would be better to use **nn.ReLU** instead of **nn.functional.relu**. To begin your workflow, a **DianaModule** instance needs to be defined from a standard pytorch Module using the **from_fp_model** class method. This instantiation defines some specifications and generates the graph we'll be using for the rest of the conversion and training processes.  
- ## Iterative Training
  - **Iteration 1**: As regularly done, the torch floating-point Model is trained. 
  - **Iteration 2**: Model is fake-quantized to 8-bits and then re-trained to regain the lost accuracy. 
  - **Iteration 3**: Model is re-quantized to diana specific parameters using **map_scale**. For example, the weights of a convolution in the analog core are mapped to terneray. After the mapping, the model is retrained again. 
  - **Iteration 4**: Scales are clipped to the closest power of 2, non-ideal analog core behaviour enabled(TBD) and model is retrained. 
- ## Graph Generation 
    After training is done, the fake-quantized model is integrized ([check true-quantization section for more info](#true-quantization) 
) and an ONNX model is generated and exported with the DORY-specific annotations 

# Model Conversion
## Structure 
The neural network conversion process in Quantlib, whether it be floating-point to fake-quantized or fake-quantized to true-quantized, is implemented as a list of graph editors that edit the operands and the operations in the computational graph generated by the torch symbolic tracer. Each editor is composed of a finder and an applier. The finder is responsible for collecting application points matching a certain criteria, while the applier is responsible for actually changing the graph or adding meta-data(annotation). 
## Fake-Quantization (Simplified)
- ### Cannonicalisation
  - Non-modular functions converter: Funcitonal torch operations like torch.functional.relu are converted to modular operation (nn.ReLU)
  - Batchnorm Folding: Bias of conv layer is absorbed into batchnorm layer 
- ### Fake quantization 
    - Module Wise Converter: mapping standing torch modules to diana-specific modules
    - Interposer: This editor is responsible for adding nodes simulating the quantization operations that happen in hardware (like adc) or ensuring that weights/activations conform to the hardware specifications=
    - Harmonise adds: This editor ensures that additions are matched (e.g, 8-bits + 8-bits)
    - Activaitons fuser: This editor fuses succeding activations to eliminate unnecessary computation. 
## True-Quantization (Simplified)
- Linear operation integrizer: This editor rewrites the diana modules back to standard pytorch modules, but with quantized parameters
- Requantizer: This editor is responsible for the requantization operations that was happening before or after a diana-specific module. 


# Quantlib Framework 


## Important properties 
### **QrangeSpecType**

property that specifies the quantisation integer ranges. The qrangespectype can be defined by a dictionary containing 2 different attributes: the quantized range and offset. The range can be defined using the key words **bitwidth** , **n_levels** , **limpbitwidth**
. The offset can be defined using the key words **offset** and **signed**. Note: the qrangespectype can also be defined directly by assigning it to **'binary'** or **'ternary'**. 
### **QGranularitySpecType**
This property defines how the statistics are collected and how scales are calculated. The granularity can be defined **'per-array'** (per layer) or **'per-outchannel_weights'** (per channel). Notice that quantized activation functions must be implemented with a granularity type of **'per-array'**
### **QHParamsInitStrategySpecType**
This property defines how the quatnization hyperparameters will be initialized after the observation. Notice that at least 1 forward pass must be completed before initialization. The currently implemented values are: **'const'**, **'minmax'**, **'meanstd'**

## Observer Class 
When initialising any quantlib class, an observer object is created. This observer object is responsible for collecting **tensor statistics** about the input, and is initialized with the **QGranularitySpecType** of the class. 
## Quantisation Operation

When defining your custom quantized classes, make sure to register your own autograd quantization function using the _register_qop function and implement it in the _call_op function. 

## Extending Quantlib 
[Check Model Conversion](#model-conversion)


Notes: 
dont use same relu module multiple times
view instead of flatten 
       
