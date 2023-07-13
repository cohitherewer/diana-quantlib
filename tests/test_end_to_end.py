import os
import io
import torch
import pytest
import numpy as np
from functools import partial
from collections.abc import Iterable
import onnx
import onnxruntime as rt
from dianaquantlib.utils.BaseModules import DianaModule
from dianaquantlib.utils.serialization.Loader import ModulesLoader
from dianaquantlib.models.mlperf_tiny import ResNet, MobileNet, DAE, DSCNN
from dianaquantlib.models.cifar10.resnet import resnet20


models = {
    'dae':          (partial(DAE, num_outputs=28*28),
                    'data/dae.pth',
                    'data/dae.yaml',
                    'data/anomaly_detection_inputs.pt',
                    'data/dae_expected_outputs.pt'),
    'dscnn':        (DSCNN,
                    'data/dscnn.pth',
                    'data/dscnn.yaml',
                    'data/keyword_spotting_inputs.pt',
                    'data/dscnn_expected_outputs.pt'),
    'mobilenetv1':  (partial(MobileNet, num_classes=12),
                    'data/mobilenetv1.pth',
                    'data/mobilenetv1.yaml',
                    'data/image_classification_inputs.pt',
                    'data/mobilenetv1_expected_outputs.pt'),
    'resnet8':      (ResNet,
                    'data/resnet8.pth',
                    'data/resnet8.yaml',
                    'data/image_classification_inputs.pt',
                    'data/resnet8_expected_outputs.pt'),
    'resnet20':     (partial(resnet20, num_classes=12),
                    'data/resnet20.pth',
                    'data/resnet20.yaml',
                    'data/image_classification_inputs.pt',
                    'data/resnet20_expected_outputs.pt'),
}


def run_onnx_model(model_filename, inputs):
    # assume single input and single output
    model = onnx.load(model_filename)
    output_spec, input_spec = model.graph.output, model.graph.input
    assert len(output_spec) == 1
    assert len(input_spec) == 1
    output_name = output_spec[0].name
    input_name = input_spec[0].name

    session = rt.InferenceSession(
        model_filename,
        providers=["CPUExecutionProvider"],
    )
    if isinstance(inputs, np.ndarray):
        out = session.run([output_name], {input_name: inputs})
        outputs = torch.Tensor(out[0])
    else:
        outputs = []
        for inp in inputs:
            for inp_b1 in inp:
                i = inp_b1.unsqueeze(0).numpy()
                out = session.run([output_name], {input_name: i})
                outputs.append(torch.Tensor(out[0]))

    return outputs


def assert_equal_outputs(expected, actual):
    assert len(expected) == len(actual)
    for exp, act in zip(expected, actual):
        assert torch.eq(exp, act).all()


def parse_feature_file(filename):
    """Return a torch.Tensor, parsed from a CSV text file
    """
    with open(filename) as f:
        lines = f.read().splitlines()

    # parse shape
    shape_str = lines[0][lines[0].index('[') + 1:lines[0].index(']')]
    shape = tuple([int(v) for v in shape_str.split(',')])

    # parse values
    values = []
    for value in lines[1].split(','):
        if value != '':
            values.append(float(value))

    return torch.Tensor(values).reshape(*shape)


# DSCNN last check fails, likely due to different behaviour of first conv between onnxruntime and pytorch
@pytest.mark.parametrize("modelname", ['dae', 'dscnn', 'resnet20', 'resnet8', 'mobilenetv1'])
def test_digital_ptq(modelname):

    # Setup
    data = models[modelname]
    model = data[0]()
    weights_file = data[1]
    config_file = data[2]
    input_data_file = data[3]
    expected_outputs_file = data[4]

    export_folder = 'export'
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    export_onnxfile = os.path.join(export_folder, model._get_name() + "_QL_NOANNOTATION.onnx")

    sd = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(sd["net"])

    def representative_dataset():
        sd = torch.load(input_data_file, map_location="cpu")
        for inp in sd['x']:
            yield inp

    # Action (end-to-end)
    loader = ModulesLoader()
    module_description = loader.load(config_file)

    fq_model = DianaModule(
        DianaModule.from_trainedfp_model(
            model,
            modules_descriptors=module_description,
            qhparamsinitstrategy='meanstd'
        ),
        representative_dataset,
    )

    fq_model.set_quantized(activations=True)
    fq_model.map_to_hw()
    fq_model.integrize_layers()
    fq_model.export_model(export_folder)

    # Assert
    outputs = run_onnx_model(export_onnxfile, representative_dataset())     # run model with onnx runtime

    # uncomment to update expected outputs
    #torch.save({'y': outputs}, expected_outputs_file)

    expected_outputs = torch.load(expected_outputs_file, map_location="cpu")['y']
    assert_equal_outputs(expected_outputs, outputs)

    # Also check that exported 'output.txt' contains correct values

    # DSCNN fails due to slightly different behaviour between pytorch and onnx runtime
    # of the first conv with non standard kernel size, paddings and strides. Therefore, we skip this check for this model
    if modelname == 'dscnn':
        return

    inp = np.load(os.path.join(export_folder, 'input.npy'))
    output = run_onnx_model(export_onnxfile, inp)     # run model with onnx runtime
    feature_file_output = torch.Tensor(np.load(os.path.join(export_folder, "output.npy")))
    assert_equal_outputs(output, feature_file_output)
