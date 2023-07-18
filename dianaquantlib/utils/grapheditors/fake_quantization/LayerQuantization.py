import torch
import torch.fx as fx
from torch import nn
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule, _QActivation
from quantlib.editing.editing.editors import Rewriter, Applier
from quantlib.editing.editing.editors.optrees import OpTree
from quantlib.editing.editing.editors.base.editor import Editor
from quantlib.editing.graphs.fx import quantlib_symbolic_trace
from quantlib.editing.editing.float2fake.quantisation.addtreeharmoniser.finder import AddTreeFinder
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.nnmodules import NNModuleDescription, Candidates, Roles, generate_named_patterns
from quantlib.editing.editing.editors.nnmodules.rewriter.factory import get_rewriter_class
from quantlib.editing.editing.editors.nnmodules.finder.nnsequential import PathGraphMatcher
from quantlib.editing.editing.editors.nnmodules.applier import NNModuleApplier
from quantlib.editing.editing.editors.nnmodules.applicationpoint import NodesMap
from dianaquantlib.core.Operations import AnalogOutIdentity, DIANAReLU, DIANAIdentity


activation_candidates = Candidates([
        ('QIdentity',  NNModuleDescription(class_=nn.Identity,  kwargs={})),
        ('QReLU',      NNModuleDescription(class_=nn.ReLU,      kwargs={})),
        ('QReLU6',     NNModuleDescription(class_=nn.ReLU6,     kwargs={})),
        ('QLeakyReLU', NNModuleDescription(class_=nn.LeakyReLU, kwargs={})),
    ])


dropout_candidates = Candidates([
        ('', None),
        ('Dropout', NNModuleDescription(class_=nn.Dropout, kwargs={})),
        ('Dropout2d', NNModuleDescription(class_=nn.Dropout2d, kwargs={})),
    ])


avgpool_roles = Roles([
    ('activation_in', activation_candidates),
    ('dropout_in', dropout_candidates),
    ('avgpool', Candidates([
        ('AdaptiveAvgPool2d', NNModuleDescription(class_=nn.AdaptiveAvgPool2d, kwargs={'output_size': (1, 1)})),
    ])),
    ('flatten', Candidates([
        ('', None),
        ('Flatten', NNModuleDescription(class_=nn.Flatten, kwargs={})),
    ])),
    ('dropout_out', dropout_candidates),
    ('activation_out', activation_candidates),
])


class AvgPoolAlignOutputScaleApplier(NNModuleApplier):

    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:
        """Ensure the quantization scale of activation_out equals that of activation_in
        """
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_activation_in = name_to_match_module['activation_in']
        module_activation_out = name_to_match_module['activation_out']

        module_activation_out.scale = module_activation_in.scale

        return g


class AvgPoolAlignOutputScale(ComposedEditor):

    def __init__(self):
        # generate rewriters
        admissible_screenplays = list(avgpool_roles.all_screenplays)

        # programmatically generate all the patterns, then for each pattern generate the corresponding `Rewriter`
        editors = []
        for name, pattern in generate_named_patterns(avgpool_roles, admissible_screenplays):
            class_name = name + 'AvgPoolAlignOutputScale'

            class_ = get_rewriter_class(class_name, pattern, PathGraphMatcher, AvgPoolAlignOutputScaleApplier)
            editors.append(class_())

        super().__init__(editors)


class AddTreeAlignInputScalesApplier(Applier):

    def _apply(self, g: fx.GraphModule, ap: OpTree, id_: str) -> fx.GraphModule:
        """Ensure both input scales of add are equal
        """
        in_modules = [g.get_submodule(n.target) for n in ap.inbound_frontier]
        user_nodes = list(ap.root.users)
        assert len(user_nodes) == 1, "TODO: currently don't know what to do with zero or multiple user nodes of add op"
        out_module = g.get_submodule(user_nodes[0].target)
        in_scales = [mod.scale for mod in in_modules]
        out_scale = out_module.scale

        # Rules for determining the scale:
        # 1) aligned value of input scales <= output scale
        #    motivation: The sum of two values is usually a larger number so it does not make sense
        #                to make both input scales > output scale.
        #                Violating this rule can also introduce an output division op with division factor < 1
        #                after integerization, resulting in an add op that cannot be performed
        #                on the digital accelerator (div factor must be >= 1)
        # 2) prefer to use the largest input scale as aligned scale value (not violating the previous rule)
        #    motivation: Prefer range over precision
        aligned_scale = torch.min(torch.concat(in_scales).max().unsqueeze(0), out_scale)

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
        self.align_avg_pool_output_scale = AvgPoolAlignOutputScale()

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
            g = self.align_avg_pool_output_scale(g)

        return g
