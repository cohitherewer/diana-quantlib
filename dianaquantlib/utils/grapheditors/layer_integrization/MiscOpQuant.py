import torch
from torch import nn
import torch.fx as fx
from quantlib.editing.editing.editors.base.editor import Editor
from quantlib.editing.editing.editors.nnmodules.applier import NNModuleApplier
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.pattern import NNSequentialPattern
from quantlib.editing.editing.editors.nnmodules.rewriter.factory import get_rewriter_class
from quantlib.editing.editing.editors.nnmodules.applicationpoint import NodesMap
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.factory import generate_named_patterns
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.roles import Roles
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.candidates import Candidates, NNModuleDescription
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.nnmodules.finder.nnsequential import PathGraphMatcher


class DianaMiscOpIntegrizerApplier(NNModuleApplier):

    def __init__(self, pattern: NNSequentialPattern):
        super().__init__(pattern)

    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:

        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_eps_in  = name_to_match_module['eps_in']
        module_eps_out = name_to_match_module['eps_out']

        # Pooling has no parameters, do just set eps_in and eps_out to 1 in order to integerize the op
        module_eps_in.set_eps_out(torch.ones_like(module_eps_in.eps_out))
        module_eps_out.set_eps_in(torch.ones_like(module_eps_out.eps_in))

        return g


eps_candidates = Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs={'eps': torch.Tensor([1.0])})),
    ])


misc_roles = Roles([
    ('eps_in', eps_candidates),
    # TODO: add more safe ops here
    ('misc', Candidates([
        ('Dropout', NNModuleDescription(class_=nn.Dropout, kwargs={})),
        ('Dropout2d', NNModuleDescription(class_=nn.Dropout2d, kwargs={})),
        ('Flatten', NNModuleDescription(class_=nn.Flatten, kwargs={})),
        ('AdaptiveAvgPool2d', NNModuleDescription(class_=nn.AdaptiveAvgPool2d, kwargs={'output_size': (1, 1)})),
    ])),
    ('eps_out', eps_candidates),
])


class DianaMiscOpIntegrizer(ComposedEditor):
    """Integrizer for miscellaneous operations that can be integerized by simply setting eps_in = eps_out = 1
    """
    def __init__(self):
        # generate rewriters
        admissible_screenplays = list(misc_roles.all_screenplays)

        # programmatically generate all the patterns, then for each pattern generate the corresponding `Rewriter`
        editors = []
        for name, pattern in generate_named_patterns(misc_roles, admissible_screenplays):
            class_name = name + 'DianaIntegeriser'

            class_ = get_rewriter_class(class_name, pattern, PathGraphMatcher, DianaMiscOpIntegrizerApplier)
            editors.append(class_())

        super().__init__(editors)
