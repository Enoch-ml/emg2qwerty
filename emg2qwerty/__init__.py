# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


##### FOR CNN + RNN HYBRID:
# from .modules import (
#     BiLSTMEncoder,
#     MultiBandRotationInvariantMLP,
#     MultiBandSpectrogramCNN,
#     SpectrogramNorm,
#     SpectrogramBandCNN,
#     TDSConvEncoder,
# )

# __all__ = [
#     "BiLSTMEncoder",
#     "MultiBandRotationInvariantMLP",
#     "MultiBandSpectrogramCNN",
#     "SpectrogramNorm",
#     "SpectrogramBandCNN",
#     "TDSConvEncoder",
# ]


##### FOR RECURRENT ARCHITECTURE (LSTM):
# from .modules import (
#     BiLSTMEncoder,
#     MultiBandRotationInvariantMLP,
#     MultiBandSpectrogramCNN,
#     RotationInvariantBiLSTMBackbone,
#     SpectrogramNorm,
#     SpectrogramBandCNN,
#     TDSConvEncoder,
# )

# __all__ = [
#     "BiLSTMEncoder",
#     "MultiBandRotationInvariantMLP",
#     "MultiBandSpectrogramCNN",
#     "RotationInvariantBiLSTMBackbone",
#     "SpectrogramNorm",
#     "SpectrogramBandCNN",
#     "TDSConvEncoder",
# ]


##### FOR CONFORMER-STYLE TRANSFORMER:
from .modules import (
    BiLSTMEncoder,
    ConformerEncoder,
    MultiBandRotationInvariantMLP,
    MultiBandSpectrogramCNN,
    RotationInvariantConformerBackbone,
    SpectrogramNorm,
    SpectrogramBandCNN,
    TDSConvEncoder,
)

__all__ = [
    "BiLSTMEncoder",
    "ConformerEncoder",
    "MultiBandRotationInvariantMLP",
    "MultiBandSpectrogramCNN",
    "RotationInvariantConformerBackbone",
    "SpectrogramNorm",
    "SpectrogramBandCNN",
    "TDSConvEncoder",
]