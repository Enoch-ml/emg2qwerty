# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)






##### FOR CNN + RNN HYBRID:
class SpectrogramBandCNN(nn.Module):
    """
    2D CNN encoder for a single band of spectrogram input.

    Input:
        (T, N, electrode_channels, freq_bins)

    Output:
        (T, N, out_features)
    """

    def __init__(
        self,
        electrode_channels: int,
        freq_bins: int,
        conv_channels: Sequence[int] = (16, 32, 64),
        cnn_out_features: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert len(conv_channels) > 0

        layers: list[nn.Module] = []
        in_channels = 1

        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[-1] * 4 * 4, cnn_out_features),
            nn.ReLU(),
        )

        self.electrode_channels = electrode_channels
        self.freq_bins = freq_bins
        self.out_features = cnn_out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (T, N, C, F)
        returns: (T, N, out_features)
        """
        T, N, C, F = inputs.shape
        assert C == self.electrode_channels
        assert F == self.freq_bins

        x = inputs.reshape(T * N, 1, C, F)   # (T*N, 1, C, F)
        x = self.conv(x)                     # (T*N, conv_channels[-1], C, F)
        x = self.pool(x)                     # (T*N, conv_channels[-1], 4, 4)
        x = self.proj(x)                     # (T*N, out_features)
        return x.reshape(T, N, self.out_features)


class MultiBandSpectrogramCNN(nn.Module):
    """
    Apply a separate 2D CNN encoder to each band.

    Input:
        (T, N, num_bands, electrode_channels, freq_bins)

    Output:
        (T, N, num_bands, cnn_out_features)
    """

    def __init__(
        self,
        electrode_channels: int,
        freq_bins: int,
        cnn_out_features: int,
        conv_channels: Sequence[int] = (16, 32, 64),
        num_bands: int = 2,
        stack_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        self.band_cnns = nn.ModuleList(
            [
                SpectrogramBandCNN(
                    electrode_channels=electrode_channels,
                    freq_bins=freq_bins,
                    conv_channels=conv_channels,
                    cnn_out_features=cnn_out_features,
                    dropout=dropout,
                )
                for _ in range(num_bands)
            ]
        )

        self.out_features = cnn_out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (T, N, num_bands, C, F)
        returns: (T, N, num_bands, out_features)
        """
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            cnn(_input) for cnn, _input in zip(self.band_cnns, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class BiLSTMEncoder(nn.Module):
    """
    BiLSTM encoder for sequence inputs of shape (T, N, input_size).

    Output:
        (T, N, hidden_size * num_directions)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.output_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (T, N, input_size)
        returns: (T, N, output_size)
        """
        x, _ = self.rnn(inputs)
        return x






##### FOR RECURRENT ARCHITECTURE (LSTM):
class RotationInvariantBiLSTMBackbone(nn.Module):
    """
    Recurrent backbone for EMG spectrogram sequences.

    Input:
        (T, N, num_bands, electrode_channels, freq_bins)

    Output:
        (T, N, rnn_output_size)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        electrode_channels: int = 16,
        num_bands: int = 2,
        rnn_hidden_size: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        self.num_bands = num_bands
        self.electrode_channels = electrode_channels

        self.norm = SpectrogramNorm(channels=num_bands * electrode_channels)

        self.frame_encoder = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            pooling=pooling,
            offsets=offsets,
            num_bands=num_bands,
        )

        rnn_input_size = num_bands * mlp_features[-1]

        self.sequence_encoder = BiLSTMEncoder(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.output_size = self.sequence_encoder.output_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (T, N, bands, C, F)
        returns: (T, N, rnn_output_size)
        """
        x = self.norm(inputs)                  # (T, N, bands, C, F)
        x = self.frame_encoder(x)             # (T, N, bands, mlp_features[-1])
        x = x.flatten(start_dim=2)            # (T, N, bands * mlp_features[-1])
        x = self.sequence_encoder(x)          # (T, N, output_size)
        return x





##### FOR CONFORMER-STYLE TRANSFORMER:
class TemporalSubsampling(nn.Module):
    """
    Temporal subsampling block for sequence inputs of shape (T, N, C).

    Applies two strided Conv1d layers over time to reduce sequence length.

    Input:
        (T, N, in_features)

    Output:
        (T_out, N, out_features)

    Notes:
        - Time is reduced by roughly a factor of 4 with stride=2 twice.
        - The exact output length is tracked by `output_lengths`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert kernel_size >= 1
        assert stride >= 1

        self.kernel_size = kernel_size
        self.stride = stride

        self.conv1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _conv_out_length(self, lengths: torch.Tensor) -> torch.Tensor:
        # PyTorch Conv1d output size formula:
        # floor((L + 2P - D*(K-1) - 1)/S + 1)
        # Here dilation=1 and padding=kernel_size//2.
        p = self.kernel_size // 2
        k = self.kernel_size
        s = self.stride
        lengths = torch.div(lengths + 2 * p - (k - 1) - 1, s, rounding_mode="floor") + 1
        lengths = torch.div(lengths + 2 * p - (k - 1) - 1, s, rounding_mode="floor") + 1
        return lengths.clamp_min(1)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return self._conv_out_length(lengths)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (T, N, C)
        returns: (T_out, N, C_out)
        """
        x = inputs.movedim(0, 1).movedim(1, 2)   # (N, C, T)
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x.movedim(2, 1).movedim(0, 1)        # (T_out, N, C_out)
        return x


class ConformerFeedForward(nn.Module):
    """
    Position-wise feedforward block used in Conformer.
    """

    def __init__(
        self,
        d_model: int,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = d_model * ff_multiplier
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class ConformerConvModule(nn.Module):
    """
    Lightweight Conformer convolution module operating on (T, N, C).

    Structure:
        LayerNorm
        -> pointwise conv expansion + GLU
        -> depthwise temporal conv
        -> BatchNorm1d
        -> SiLU
        -> pointwise conv projection
        -> Dropout
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for same padding"

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(inputs)              # (T, N, C)
        x = x.movedim(0, 1).movedim(1, 2)        # (N, C, T)
        x = self.pointwise_in(x)                 # (N, 2C, T)
        x = self.glu(x)                          # (N, C, T)
        x = self.depthwise(x)                    # (N, C, T)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.pointwise_out(x)
        x = self.dropout(x)
        x = x.movedim(2, 1).movedim(0, 1)        # (T, N, C)
        return x


class ConformerBlock(nn.Module):
    """
    Standard Conformer-style encoder block for sequence input (T, N, d_model).

    Structure:
        x = x + 0.5 * FFN(x)
        x = x + MHSA(LN(x))
        x = x + ConvModule(x)
        x = x + 0.5 * FFN(x)
        x = LayerNorm(x)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_multiplier: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.ffn1 = ConformerFeedForward(
            d_model=d_model,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
        )

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.self_attn_dropout = nn.Dropout(dropout)

        self.conv_module = ConformerConvModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.ffn2 = ConformerFeedForward(
            d_model=d_model,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = inputs

        x = x + 0.5 * self.ffn1(x)

        x_attn = self.self_attn_norm(x)
        x_attn, _ = self.self_attn(
            x_attn,
            x_attn,
            x_attn,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.self_attn_dropout(x_attn)

        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)

        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    """
    Stack of Conformer blocks for inputs of shape (T, N, d_model).
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 8,
        num_heads: int = 4,
        ff_multiplier: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert num_layers > 0

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def make_key_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """
        Returns mask of shape (N, T) where True indicates padding positions.
        """
        device = lengths.device
        t = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
        return t >= lengths.unsqueeze(1)                       # (N, T)

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = inputs
        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = self.make_key_padding_mask(lengths, x.shape[0])

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        return x


class RotationInvariantConformerBackbone(nn.Module):
    """
    EMG frontend + temporal subsampling + Conformer encoder.

    Input:
        (T, N, num_bands, electrode_channels, freq_bins)

    Output:
        (T_out, N, d_model)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        d_model: int = 256,
        num_layers: int = 8,
        num_heads: int = 4,
        ff_multiplier: int = 4,
        conv_kernel_size: int = 15,
        subsampling_kernel_size: int = 3,
        subsampling_stride: int = 2,
        electrode_channels: int = 16,
        num_bands: int = 2,
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_bands = num_bands
        self.electrode_channels = electrode_channels

        self.norm = SpectrogramNorm(channels=num_bands * electrode_channels)

        self.frame_encoder = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            pooling=pooling,
            offsets=offsets,
            num_bands=num_bands,
        )

        frontend_dim = num_bands * mlp_features[-1]

        self.frontend_proj = nn.Sequential(
            nn.Linear(frontend_dim, d_model),
            nn.Dropout(dropout),
        )

        self.subsampling = TemporalSubsampling(
            in_features=d_model,
            out_features=d_model,
            kernel_size=subsampling_kernel_size,
            stride=subsampling_stride,
            dropout=dropout,
        )

        self.encoder = ConformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.output_size = d_model

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return self.subsampling.output_lengths(input_lengths)

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        inputs: (T, N, bands, C, F)
        input_lengths: (N,)
        returns:
            emissions_input: (T_out, N, d_model)
            output_lengths: (N,) or None
        """
        x = self.norm(inputs)                   # (T, N, bands, C, F)
        x = self.frame_encoder(x)              # (T, N, bands, mlp_features[-1])
        x = x.flatten(start_dim=2)             # (T, N, bands * mlp_features[-1])
        x = self.frontend_proj(x)              # (T, N, d_model)
        x = self.subsampling(x)                # (T_out, N, d_model)

        output_lengths = None
        if input_lengths is not None:
            output_lengths = self.output_lengths(input_lengths)

        x = self.encoder(x, lengths=output_lengths)
        return x, output_lengths