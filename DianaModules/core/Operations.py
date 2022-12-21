from abc import abstractmethod
import enum
from random import randint
from tokenize import group
from turtle import forward
import torch
from torch import Tensor, nn
from typing import Union, Tuple


from quantlib.algorithms.qalgorithms.qatalgorithms.pact.qactivations import (
    PACTReLU,
)
from quantlib.algorithms.qbase.qhparams.qhparams import create_qhparams
from quantlib.algorithms.qbase.qrange.qrange import resolve_qrangespec
from quantlib.algorithms.qalgorithms.qatalgorithms.pact import PACTConv2d


from quantlib.algorithms.qmodules.qmodules import QIdentity


from quantlib.algorithms.qmodules.qmodules.qlinears import QConv2d, QLinear

import math

from quantlib.algorithms.qbase import (
    QRangeSpecType,
    QGranularitySpecType,
    QHParamsInitStrategySpecType,
)
from DianaModules.utils._FakeQuantizer import _FakeDQuantiser, _FakeAQuantiser

from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
from quantlib.algorithms.qbase import get_zero_scale, get_scale


class DianaBaseOperation:
    @abstractmethod
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        pass

    def redefine_qhparams(self: _QModule, qrangespec: QRangeSpecType):
        assert issubclass(type(self), _QModule)
        device = self.zero.device
        self._qrange = resolve_qrangespec(qrangespec)
        zero, n_levels, step, scale = create_qhparams(self._qrange)
        self.zero = torch.tile(zero, self.zero.shape)
        self.n_levels = torch.tile(n_levels, self.n_levels.shape)
        self.step = torch.tile(step, self.step.shape)
        self.scale = torch.tile(scale, self.scale.shape)
        if self._pin_offset:
            scale = get_scale(
                self.min_float,
                self.max_float,
                self.zero,
                self.n_levels,
                self.step,
            )
            self.scale.data.copy_(scale)
        else:
            zero, scale = get_zero_scale(
                self.min_float, self.max_float, self.n_levels, self.step
            )
            self.zero.data.copy_(zero)
            self.scale.data.copy_(scale)
        self._set_clipping_bounds()

    def define_bitwidth_clipping(self):
        if not hasattr(self, "bw_clip_hi"):
            self.register_buffer(
                "bw_clip_lo",
                torch.exp2(
                    torch.round(
                        torch.log2(
                            (
                                torch.abs(self.clip_lo)
                                / (self.scale * self.step)
                            )
                        )
                    )
                ),
            )
            self.register_buffer(
                "bw_clip_hi",
                torch.exp2(
                    torch.round(
                        torch.log2(
                            (
                                torch.abs(self.clip_hi)
                                / (self.scale * self.step)
                            )
                        )
                    )
                )
                - 1,
            )
        else:
            self.bw_clip_lo = torch.exp2(
                torch.round(
                    torch.log2(
                        (torch.abs(self.clip_lo) / (self.scale * self.step))
                    )
                )
            )  # quantized clip lo and high
            self.bw_clip_hi = (
                torch.exp2(
                    torch.round(
                        torch.log2(
                            (
                                torch.abs(self.clip_hi)
                                / (self.scale * self.step)
                            )
                        )
                    )
                )
                - 1
            )

        mask = self.clip_lo < 0
        # if self.clip_lo < 0: # TODO  for now just leave this so low clipping bound is -127  , implement this later specifically for simd model
        self.bw_clip_lo[mask] = -self.bw_clip_lo[mask]  # + 1


class IdentityType(enum.Enum):
    default = enum.auto()  # 8 Bits
    AIMC_IN = enum.auto()  # 7 bits
    AIMC_OUT = enum.auto()  # 6 bits
    DIGITAL_IN = enum.auto()  # 8 bits (digital core)
    DIGITAL_OUT = enum.auto()  # 8 bits


class DIANALinear(QLinear, DianaBaseOperation):
    def __init__(
        self,
        qrangespec: QRangeSpecType,
        qgranularityspec: QGranularitySpecType,
        qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
        in_features: int,
        out_features: int,
        bias=False,
    ):
        super().__init__(
            qrangespec,
            qgranularityspec,
            qhparamsinitstrategyspec,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def _register_qop(
        self,
    ):  # used for autograd functions with non-standard backward gradients
        self._qop = _FakeDQuantiser.apply

    def stop_observing(self):
        super().stop_observing()
        self.define_bitwidth_clipping()

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._qop(
            x, self.bw_clip_lo, self.bw_clip_hi, self.step, self.scale
        )

    def forward(self, x: torch.Tensor):
        return super().forward(x)


# Activations
class DIANAIdentity(
    QIdentity, DianaBaseOperation
):  # general purpose identity for harmoniser adds ( Quant operation )
    def __init__(
        self,
        qrangespec: Union[QRangeSpecType, IdentityType],
        qgranularityspec: QGranularitySpecType,
        qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
    ):
        if type(qrangespec) == IdentityType:
            if qrangespec == IdentityType.AIMC_IN:
                qrangespec = {"bitwidth": 7, "signed": True}
            elif qrangespec == IdentityType.AIMC_OUT:
                qrangespec = {"bitwidth": 6, "signed": True}
            else:
                qrangespec = {"bitwidth": 8, "signed": True}

        super().__init__(
            qrangespec, qgranularityspec, qhparamsinitstrategyspec
        )

    def _register_qop(
        self,
    ):  # used for autograd functions with non-standard backward gradients
        self._qop = _FakeDQuantiser.apply

    def stop_observing(self):
        super().stop_observing()
        self.define_bitwidth_clipping()

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        return self._qop(
            x, self.bw_clip_lo, self.bw_clip_hi, self.step, self.scale
        )


class DIANAReLU(PACTReLU, DianaBaseOperation):
    def __init__(
        self,
        qrangespec: QRangeSpecType,
        qgranularityspec: QGranularitySpecType,
        qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
        inplace: bool = False,
    ):

        super().__init__(
            qrangespec, qgranularityspec, qhparamsinitstrategyspec, inplace
        )

    def call_qop(self, x: torch.Tensor) -> torch.Tensor:
        return super().call_qop(x)

    def freeze(self):
        self.clip_lo.requires_grad = False
        self.clip_hi.requires_grad = False
        self._clipping_bounds_are_frozen |= True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self._is_quantised & True == True:
            self.scale = torch.exp2(torch.round(torch.log2(self.scale)))
        return out


# How I have it right now it will be a convolution in the digital core if it's not followed by a batch norm otherwise it's an analog core
class DIANAConv2d(QConv2d, DianaBaseOperation):  # digital core
    def __init__(
        self,
        qrangespec: QRangeSpecType,
        qgranularityspec: QGranularitySpecType,
        qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...] = 1,
        padding: int = 0,
        dilation: Tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        self.is_analog = (
            not bias
        )  # In linearopbn cannonocalisation the bias is set to none if bn follows conv

        super().__init__(
            qrangespec,
            qgranularityspec,
            qhparamsinitstrategyspec,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        )

    def stop_observing(self):
        super().stop_observing()
        self.define_bitwidth_clipping()

    def _register_qop(
        self,
    ):  # used for autograd functions with non-standard backward gradients
        if self.is_analog:
            self._qop = _FakeAQuantiser.apply
        else:
            self._qop = _FakeDQuantiser.apply

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:

        return self._qop(
            x, self.bw_clip_lo, self.bw_clip_hi, self.step, self.scale
        )

    def forward(self, x: torch.Tensor):
        # round bias here
        return super().forward(x)


# Analog Core Conv Operation: DAC ,  ,  split channels conv ,  quantize ,noise then accumulate
class AnalogConv2d(DIANAConv2d):
    def __init__(
        self,
        qrangespec: QRangeSpecType,
        qgranularityspec: QGranularitySpecType,
        qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Tuple[int, ...] = 1,
        padding: str = 0,
        dilation: Tuple[int, ...] = 1,
        groups: int = 1,
        array_size: int = 1152,
    ):

        if isinstance(kernel_size, Tuple):
            k_size = kernel_size[0] * kernel_size[1]
        else:
            k_size = kernel_size**2
        self.max_channels = math.floor(array_size / k_size)
        assert (
            self.max_channels >= 1
        ), f"Array size must be at least kernel_size * kernel_size: {kernel_size**2}"
        # assert groups ==1 , f"For grouped convolutions, please run on the digital core. "
        super().__init__(
            qrangespec,
            qgranularityspec,
            qhparamsinitstrategyspec,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,
            bias=False,
        )

    def forward(self, x: torch.Tensor):  # returns a five dimensionsal tensor
        group_count = (
            1
            if self.max_channels >= x.size(1)
            else math.ceil(x.size(1) / self.max_channels)
        )
        counter = x.size(1)

        tile_size = x.size(1) if group_count <= 1 else self.max_channels
        padded_out = None
        if self._is_quantised:
            weight = self.qweight
        else:
            weight = self.weight
        min_i = 0
        max_i = 0
        for i in range(
            group_count
        ):  # chance for parallelization across distributed loads

            max_i = min(tile_size + max_i, x.size(1))

            group_passed = x[:, min_i:max_i, :, :]
            pass_weight = weight[:, min_i:max_i, :, :]

            conv_out = nn.functional.conv2d(
                group_passed,
                pass_weight,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            if padded_out is None:
                padded_out = torch.zeros(
                    group_count,
                    x.size(0),
                    self.out_channels,
                    conv_out.size(2),
                    conv_out.size(3),
                ).to(self.weight.device)
            padded_out[i] = conv_out.clone().to(padded_out.device)
            counter = counter - tile_size
            min_i = max_i
            if counter < self.max_channels:
                tile_size = counter  # remaining

        return padded_out

    def stop_observing(self):

        super(DIANAConv2d, self).stop_observing()
        if self.n_levels <= 3:
            self.register_buffer(
                "bw_clip_lo",
                torch.tile(torch.Tensor([-1]), self.clip_lo.shape),
            )
            self.register_buffer(
                "bw_clip_hi", torch.tile(torch.Tensor([1]), self.clip_hi.shape)
            )
        else:
            self.define_bitwidth_clipping()


class AnalogOutIdentity(
    DIANAIdentity
):  ## when observing , each forward pass randomly sample a group (except the last one ) and observe it
    def __init__(
        self,
        qgranularityspec: QGranularitySpecType,
        qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
    ):
        super().__init__(
            qrangespec=IdentityType.AIMC_OUT,
            qgranularityspec=qgranularityspec,
            qhparamsinitstrategyspec=qhparamsinitstrategyspec,
        )

    # def forward(self, x : torch.Tensor):
    #    if self._is_observing:
    #        with torch.no_grad():
    #
    #            i = randint(0, x.size(0) -1 )
    #            self._observer.update(x[i])

    #    if self._is_quantised:
    #        x = self._call_qop(x)
    #    else:
    #        x = super(_QModule, self).forward(x)

    #    return x


# (Partial sum )
class AnalogAccumulator(
    nn.Module
):  # given a five dimensional tensor return a reduced accumulated 4d tensor
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.sum(x, 0)


class AnalogGaussianNoise(nn.Module):  # applying noise to adc out
    def __init__(self, signed, bitwidth, SNR=25, std=1):
        super().__init__()
        range = 2**bitwidth
        self.signed = signed
        self.SNR_i = 1 / SNR
        self.sigma = std

        if signed:
            self.register_buffer(
                "clip_lo", torch.Tensor([-(2 ** (bitwidth - 1))])
            )
            self.register_buffer("clip_hi", -self.clip_lo - 1)
        else:
            self.register_buffer("clip_lo", torch.Tensor([0]))
            self.register_buffer("clip_hi", torch.Tensor([2**bitwidth]) - 1)
        self.enabled = False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def forward(self, x: torch.Tensor):
        if self.enabled:
            x = torch.round(
                x + self.SNR_i * x * (torch.randn_like(x) * self.sigma)
            )
            return torch.clamp(x, self.clip_lo, self.clip_hi)
        return x


# def backward(self , grad_in : torch.Tensor) :
#     return grad_in
