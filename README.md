# diana-training-fmw
repository for the training framework of diana


# Modules 
## Digital 

## Analog 

## Composite Blocks

# Quantlib Framework 
## Installation 
 

    pip install ./quantlib

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
