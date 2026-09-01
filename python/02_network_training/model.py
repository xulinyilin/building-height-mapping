"""Dual-branch CNN reproduced from the supplied training code and figure."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


PATCH_SIZE = 35


@dataclass(frozen=True)
class BranchConfig:
    """Definition of one valid-convolution branch."""

    in_channels: int
    channels: tuple[int, ...]
    kernels: tuple[int, ...]
    bias: bool = False

    def validate(self, patch_size: int = PATCH_SIZE) -> None:
        if len(self.kernels) != len(self.channels) or not self.kernels:
            raise ValueError("kernels and channels must be non-empty and equally long")
        if any(kernel < 1 or kernel % 2 == 0 for kernel in self.kernels):
            raise ValueError("All convolution kernels must be positive odd integers")
        if any(channel < 1 for channel in self.channels):
            raise ValueError("All output-channel counts must be positive")
        output_size = patch_size - sum(kernel - 1 for kernel in self.kernels)
        if output_size != 1:
            raise ValueError(
                f"This architecture must reduce {patch_size} x {patch_size} to 1 x 1; "
                f"the configured kernels produce {output_size} x {output_size}"
            )


def tensor_center(tensor: torch.Tensor, size: tuple[int, int] = (35, 35)) -> torch.Tensor:
    """Return the centred odd-sized crop used by the supplied implementation."""
    if tensor.ndim != 4:
        raise ValueError("Input tensor must have shape [batch, channels, height, width]")
    wanted_rows, wanted_columns = size
    if wanted_rows % 2 == 0 or wanted_columns % 2 == 0:
        raise ValueError("Centre-crop dimensions must be odd")
    rows, columns = tensor.shape[-2:]
    if rows < wanted_rows or columns < wanted_columns:
        raise ValueError(
            f"Cannot take a {wanted_rows} x {wanted_columns} crop from {rows} x {columns}"
        )
    row = rows // 2
    column = columns // 2
    half_rows = wanted_rows // 2
    half_columns = wanted_columns // 2
    return tensor[
        :,
        :,
        row - half_rows : row + half_rows + 1,
        column - half_columns : column + half_columns + 1,
    ]


class NetBranch(nn.Module):
    """A stack of unpadded Conv2d + ReLU layers, with no pooling or BatchNorm."""

    def __init__(self, config: BranchConfig) -> None:
        super().__init__()
        config.validate()
        layers: list[nn.Module] = []
        input_channels = config.in_channels
        for output_channels, kernel_size in zip(config.channels, config.kernels):
            layers.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    padding=0,
                    bias=config.bias,
                )
            )
            layers.append(nn.ReLU())
            input_channels = output_channels
        self.layers = nn.Sequential(*layers)
        self.output_dim = config.channels[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DualBranchHeightCNN(nn.Module):
    """Exact two-branch CNN; its reusable representation is the final 32-D conv output."""

    def __init__(
        self,
        ms_config: BranchConfig,
        rd_config: BranchConfig,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        ms_config.validate()
        rd_config.validate()
        self.ms_in_channels = ms_config.in_channels
        self.rd_in_channels = rd_config.in_channels
        self.ms_branch = NetBranch(ms_config)
        self.rd_branch = NetBranch(rd_config)
        self.feature_dim = self.ms_branch.output_dim + self.rd_branch.output_dim
        if self.feature_dim != 32:
            raise ValueError(
                f"The supplied regression head expects 32 features, got {self.feature_dim}"
            )

        self.fc1 = nn.Linear(32, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def encode_maps(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both final convolution maps, each shaped [B, 16, 1, 1]."""
        expected_channels = self.ms_in_channels + self.rd_in_channels
        if inputs.ndim != 4 or inputs.shape[1] != expected_channels:
            raise ValueError(
                f"Expected [B, {expected_channels}, H, W], got {tuple(inputs.shape)}"
            )
        ms_inputs = tensor_center(inputs[:, : self.ms_in_channels], (35, 35))
        rd_inputs = tensor_center(inputs[:, self.ms_in_channels :], (35, 35))
        return self.ms_branch(ms_inputs), self.rd_branch(rd_inputs)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten and concatenate the last convolution layers into 32 features."""
        ms_map, rd_map = self.encode_maps(inputs)
        ms_features = torch.flatten(ms_map, start_dim=1)
        rd_features = torch.flatten(rd_map, start_dim=1)
        deep_features = torch.cat((ms_features, rd_features), dim=1)
        return deep_features, ms_features, rd_features

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        deep_features, ms_features, rd_features = self.encode(inputs)
        height = torch.relu(self.fc1(deep_features))
        height = self.dropout(height)
        height = torch.relu(self.fc2(height))
        height = self.fc3(height).squeeze(1)
        return {
            "height": height,
            "deep_features": deep_features,
            "ms_features": ms_features,
            "rd_features": rd_features,
        }


def supplied_model(dropout: float = 0.3) -> DualBranchHeightCNN:
    """Build the exact architecture shown in the figure and attached code."""
    return DualBranchHeightCNN(
        ms_config=BranchConfig(
            in_channels=4,
            channels=(4, 4, 4, 8, 8, 16, 16, 16),
            kernels=(3, 5, 7, 9, 7, 5, 3, 3),
            bias=False,
        ),
        rd_config=BranchConfig(
            in_channels=2,
            channels=(8, 8, 8, 8, 8, 16, 16),
            kernels=(3, 5, 7, 9, 7, 7, 3),
            bias=False,
        ),
        dropout=dropout,
    )
