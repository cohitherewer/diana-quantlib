import torch
import torch.fx as fx
from torch import nn
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule, _QActivation
from quantlib.editing.editing.editors.base.editor import Editor
from dianaquantlib.core.Operations import AnalogOutIdentity, DIANAReLU, DIANAIdentity


class LayerQuantizer(Editor):
    """Set layers to quantized state and estimate the quantization parameters based on statistical observations
    on a representative dataset (calibration dataset).
    """
    def __init__(self, representative_dataset, activations=False) -> None:
        """
        representative_dataset: generator function for obtaining dataset samples for calibrating the quantization params.
        activations:            If True, also quantize activation layers.
        """
        super().__init__()
        self.representative_dataset = representative_dataset
        self.activations = activations

    def start_observing(self, g):
        """Enable statistical observers
        """
        for module in g.modules():
            if not isinstance(module, _QModule):
                continue
            if isinstance(module, _QActivation) and not self.activations:
                continue
            module.start_observing()

    def stop_observing(self, g):
        """Disable statistical observers and estimate quantization params based on observations
        """
        for module in g.modules():
            if not isinstance(module, _QModule):
                continue
            if isinstance(module, _QActivation) and not self.activations:
                continue
            module.stop_observing()

    def freeze_clipping_bound(self, g):
        """Stop modifying clip bounds in DIANAReLU during forward
        """
        if not self.activations:
            return

        for module in g.modules():
            if isinstance(module, DIANAReLU):
                module.freeze()

    def convert_scales_to_pot(self, g):
        """Convert quantization scales to power-of-two scales
        """
        # round scales to power-of-two scales
        for module in g.modules():
            if not isinstance(module, _QModule):
                continue
            if isinstance(module, _QActivation) and not self.activations:
                continue
            if isinstance(module, AnalogOutIdentity):
                continue

            # if the number of levels is high enough, we prefer range over step size and use ceil instead of round
            round_op = torch.round if module.n_levels < 128 else torch.ceil
            module.scale = torch.Tensor(torch.exp2(round_op(torch.log2(module.scale))))

    def align_ews_input_scales(self, g):
        """Ensure that the quantization scales of both inputs of an element-wise-add are the same.
        This is needed since the digital add op cannot scale both inputs individually.
        """
        if not self.activations:
            return

        for name, module in g.named_modules():
            # we match by name since harmonized add modules is traced as an nn.Module
            if 'AddTreeHarmoniser' not in name or '_input_qmodules' not in name:
                continue

            # TODO: if only one item in scales due to skip without ops, ensure that the scale is the
            # same as the dominating op (ReLU in ResNet).
            # -> make a rewriter transform for this
            scales = []
            for qm in module.modules():
                if isinstance(qm, DIANAIdentity):
                    scales.append(qm.scale)

            new_scale = torch.concat(scales).max().unsqueeze(0)
            for qm in module.modules():
                if isinstance(qm, DIANAIdentity):
                    qm.scale = new_scale

    def apply(self, g: fx.graph_module.GraphModule, *args, **kwargs) -> fx.graph_module.GraphModule:
        """Quantize all layers
        """

        self.start_observing(g)

        if self.activations:
            for data in self.representative_dataset():
                _ = g(data)
        else:
            # for estimating only the quantization parameters for the weights, a single forward sample will do
            _ = g(next(self.representative_dataset()))

        self.stop_observing(g)          # sets layers to quantized state
        self.freeze_clipping_bound(g)
        self.convert_scales_to_pot(g)   # Convert scales to power-of-two

        self.align_ews_input_scales(g)

        return g
