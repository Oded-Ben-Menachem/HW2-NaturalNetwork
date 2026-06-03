import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
            self,
            in_size,
            out_classes: int,
            channels: Sequence[int],
            pool_every: int,
            hidden_dims: Sequence[int],
            conv_params: dict = {},
            activation_type: str = "relu",
            activation_params: dict = {},
            pooling_type: str = "max",
            pooling_params: dict = {},
    ):
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        current_channels = in_channels

        for i, out_channels in enumerate(self.channels):
            # 1. Add Conv layer
            layers.append(
                nn.Conv2d(current_channels, out_channels, **self.conv_params)
            )

            # 2. Add Activation layer
            activation_cls = ACTIVATIONS[self.activation_type]
            layers.append(activation_cls(**self.activation_params))

            # 3. Add Pooling layer (only after P layers)
            if (i + 1) % self.pool_every == 0:
                pooling_cls = POOLINGS[self.pooling_type]
                layers.append(pooling_cls(**self.pooling_params))

            current_channels = out_channels

        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        """
        dummy_input = torch.zeros(1, *self.in_size)

        # Switch to eval mode and use no_grad to prevent updating
        # tracking gradients or batchnorm stats with our dummy zeros!
        was_training = self.feature_extractor.training
        self.feature_extractor.eval()

        with torch.no_grad():
            dummy_output = self.feature_extractor(dummy_input)

        self.feature_extractor.train(was_training)

        return dummy_output.numel() // dummy_output.size(0)

    def _make_mlp(self):
        in_dim = self._n_features()
        dims = list(self.hidden_dims) + [self.out_classes]

        # We explicitly instantiate the activations for the MLP so that
        # activation_params (like negative_slope) are passed correctly!
        nonlins = []
        for _ in self.hidden_dims:
            nonlins.append(ACTIVATIONS[self.activation_type](**self.activation_params))
        nonlins.append("none")

        mlp = MLP(in_dim=in_dim, dims=dims, nonlins=nonlins)
        return mlp

    def forward(self, x: Tensor):
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)
        out = self.mlp(features_flat)
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
            self,
            in_channels: int,
            channels: Sequence[int],
            kernel_sizes: Sequence[int],
            batchnorm: bool = False,
            dropout: float = 0.0,
            activation_type: str = "relu",
            activation_params: dict = {},
            **kwargs,
    ):
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # ====== YOUR CODE: ======
        main_layers = []
        curr_channels = in_channels

        for i, (out_channels, ks) in enumerate(zip(channels, kernel_sizes)):
            # Add convolutional layer (padding to preserve spatial dimensions)
            main_layers.append(
                nn.Conv2d(curr_channels, out_channels, kernel_size=ks, padding=ks // 2, bias=True)
            )

            # If it's NOT the last convolution, add dropout, batchnorm, and activation
            if i < len(channels) - 1:
                if dropout > 0:
                    main_layers.append(nn.Dropout2d(dropout))
                if batchnorm:
                    main_layers.append(nn.BatchNorm2d(out_channels))
                main_layers.append(ACTIVATIONS[activation_type](**activation_params))

            curr_channels = out_channels

        self.main_path = nn.Sequential(*main_layers)

        # Shortcut path
        if in_channels != channels[-1]:
            # Project to match the output channels of the main path
            self.shortcut_path = nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False)
        else:
            # Identity mapping
            self.shortcut_path = nn.Identity()
        # ========================

    def forward(self, x: Tensor):
        out: Tensor = None
        # ====== YOUR CODE: ======
        # Main path + Shortcut path
        out = self.main_path(x) + self.shortcut_path(x)
        # ========================
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
            self,
            in_out_channels: int,
            inner_channels: Sequence[int],
            inner_kernel_sizes: Sequence[int],
            **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. NOT the outer projections)
            The length determines the number of convolutions, EXCLUDING the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->5->10.
            The first and last arrows are the 1X1 projection convolutions,
            and the middle one is the inner convolution (corresponding to the kernel size listed in "inner kernel sizes").
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        # ====== YOUR CODE: ======
        # The first 1x1 conv projects in_out_channels -> inner_channels[0]
        # The inner convs project inner_channels[i-1] -> inner_channels[i]
        # The last 1x1 conv projects inner_channels[-1] -> in_out_channels

        # Build the full list of output channels for the convolutions
        # Notice we don't repeat the first inner_channel, we just prepend it.
        # Example: in_out=256, inner=[64].
        # Full channels should be: [64, 64, 256].
        full_channels = [inner_channels[0]] + list(inner_channels) + [in_out_channels]

        # Build the full list of kernel sizes
        # The first and last are 1x1 projections.
        full_kernel_sizes = [1] + list(inner_kernel_sizes) + [1]

        super().__init__(
            in_channels=in_out_channels,
            channels=full_channels,
            kernel_sizes=full_kernel_sizes,
            **kwargs
        )
        # ========================


class ResNet(CNN):
    def __init__(
            self,
            in_size,
            out_classes,
            channels,
            pool_every,
            hidden_dims,
            batchnorm=False,
            dropout=0.0,
            bottleneck: bool = False,
            **kwargs,
    ):
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # ====== YOUR CODE: ======
        current_channels = in_channels
        N = len(self.channels)
        P = self.pool_every

        for i in range(0, N, P):
            chunk = self.channels[i: i + P]

            if self.bottleneck:
                # בבוטלנק ה-inner channels צריכים להיות ערוץ אחד שקטן מ-in_out_channels
                # נשתמש ב-chunk[0] אבל נצמצם אותו פי 4 (מינימום 1)
                inner_dim = max(chunk[0] // 4, 1)
                layers.append(ResidualBottleneckBlock(
                    in_out_channels=current_channels,
                    inner_channels=[inner_dim],
                    inner_kernel_sizes=[3],
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    activation_type=self.activation_type,
                    activation_params=self.activation_params
                ))
            else:
                kernel_sizes = [3] * len(chunk)
                layers.append(ResidualBlock(
                    in_channels=current_channels,
                    channels=chunk,
                    kernel_sizes=kernel_sizes,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    activation_type=self.activation_type,
                    activation_params=self.activation_params
                ))

            current_channels = chunk[-1]

            pooling_cls = POOLINGS[self.pooling_type]
            layers.append(pooling_cls(**self.pooling_params))

        # ========================
        seq = nn.Sequential(*layers)
        return seq