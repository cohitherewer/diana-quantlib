import torch
import torch.fx as fx
from torch import nn
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule, _QActivation
from quantlib.editing.editing.editors import Rewriter, Applier
from quantlib.editing.editing.editors.optrees import OpTree
from quantlib.editing.editing.editors.base.editor import Editor
from quantlib.editing.graphs.fx import quantlib_symbolic_trace
from quantlib.editing.editing.float2fake.quantisation.addtreeharmoniser.finder import AddTreeFinder
from dianaquantlib.core.Operations import AnalogOutIdentity, DIANAReLU, DIANAIdentity


class AddTreeAlignInputScalesApplier(Applier):

    def _apply(self, g: fx.GraphModule, ap: OpTree, id_: str) -> fx.GraphModule:
        """Ensure both input scales of add are equal
        """
        in_modules = [g.get_submodule(n.target) for n in ap.inbound_frontier]
        scales = [mod.scale for mod in in_modules]
        aligned_scale = torch.concat(scales).max().unsqueeze(0)
        for mod in in_modules:
            mod.scale = aligned_scale

        return g


class AddTreeAlignInputScales(Rewriter):

    def __init__(self):

        finder = AddTreeFinder()
        applier = AddTreeAlignInputScalesApplier()

        super().__init__('AddTreeAlignInputScales', quantlib_symbolic_trace, finder, applier)


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
        self.align_add_input_scales = AddTreeAlignInputScales()

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

        if self.activations:
            g = self.align_add_input_scales(g)

        return g
