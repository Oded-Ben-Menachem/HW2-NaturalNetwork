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
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
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
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        curr_channels = in_channels
        for i, out_channels in enumerate(self.channels):
            # 1. Add Conv layer
            conv_kwargs = dict(self.conv_params)
            if "padding" not in conv_kwargs:
                ks = conv_kwargs.get("kernel_size", 3)
                conv_kwargs["padding"] = ks // 2 if isinstance(ks, int) else tuple(k // 2 for k in ks)
                
            layers.append(nn.Conv2d(curr_channels, out_channels, **conv_kwargs))
            
            # 2. Add Activation layer
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
            
            # 3. Add Pooling layer (only after P convolutions)
            if (i + 1) % self.pool_every == 0:
                layers.append(POOLINGS[self.pooling_type](**self.pooling_params))
                
            curr_channels = out_channels
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            dummy_input = torch.zeros(1, *self.in_size)
            was_training = self.feature_extractor.training
            self.feature_extractor.eval()
            
            with torch.no_grad():
                dummy_output = self.feature_extractor(dummy_input)
                
            self.feature_extractor.train(was_training)
            return dummy_output.numel() // dummy_output.size(0)
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        # TODO:
        #  - Create the MLP part of the model: (FC -> ACT)*M -> Linear
        #  - Use the the MLP implementation from Part 1.
        #  - The first Linear layer should have an input dim of equal to the number of
        #    convolutional features extracted by the convolutional layers.
        #  - The last Linear layer should have an output dim of out_classes.
        mlp: MLP = None
        # ====== YOUR CODE: ======
        in_dim = self._n_features()
        dims = list(self.hidden_dims) + [self.out_classes]
        
        nonlins = []
        for _ in self.hidden_dims:
            nonlins.append(ACTIVATIONS[self.activation_type](**self.activation_params))
        nonlins.append("none")
        
        mlp = MLP(in_dim=in_dim, dims=dims, nonlins=nonlins)
        # ========================
        return mlp

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        out: Tensor = None
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)
        out = self.mlp(features_flat)
        # ========================
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
            # Always use bias in the main_path conv layers
            main_layers.append(
                nn.Conv2d(curr_channels, out_channels, kernel_size=ks, padding=ks // 2, bias=True)
            )

            # Apply Dropout, BatchNorm, and Activation ONLY if it's not the last layer
            if i < len(channels) - 1:
                if dropout > 0:
                    main_layers.append(nn.Dropout2d(dropout))
                
                if batchnorm:
                    main_layers.append(nn.BatchNorm2d(out_channels))
                    
                main_layers.append(ACTIVATIONS[activation_type](**activation_params))

            curr_channels = out_channels

        self.main_path = nn.Sequential(*main_layers)

        # Create shortcut path
        if in_channels != channels[-1]:
            # No bias in the skips
            self.shortcut_path = nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False)
        else:
            self.shortcut_path = nn.Identity()
            
        self.final_activation = ACTIVATIONS[activation_type](**activation_params)
        # ========================

    def forward(self, x: Tensor):
        out: Tensor = None
        # ====== YOUR CODE: ======
        out = self.main_path(x) + self.shortcut_path(x)
        out = self.final_activation(out)
        return out
        # ========================


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
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        # ====== YOUR CODE: ======
        # Bottleneck architecture defined exactly as the assignment requires
        channels = [inner_channels[0]] + list(inner_channels) + [in_out_channels]
        kernel_sizes = [1] + list(inner_kernel_sizes) + [1]
        
        in_channels = kwargs.pop('in_channels', in_out_channels)
        
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            kernel_sizes=kernel_sizes,
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
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions (make sure to use the right stride and padding).
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        #    Reminder: the number of convolutions performed in the bottleneck block is:
        #    2 + len(inner_channels). [1 for each 1X1 proection convolution] + [# inner convolutions].
       # ====== YOUR CODE: ======
        curr_channels = in_channels
        P = self.pool_every
        N = len(self.channels)
        
        for i in range(0, N, P):
            chunk = self.channels[i:i+P]
            
            # תנאי קריטי: Bottleneck נבנה רק אם הקלט שווה לפלט (כמו ב-ResNet המקורי)
            # ואם ה-Chunk בגודל P. זה יגרום לבלוק הראשון להיווצר כ-ResidualBlock רגיל.
            if self.bottleneck and len(chunk) == P and curr_channels == chunk[-1]:
                inner_channels = chunk[1:-1]
                block = ResidualBottleneckBlock(
                    in_out_channels=chunk[-1],
                    inner_channels=inner_channels,
                    inner_kernel_sizes=[3] * len(inner_channels),
                    in_channels=curr_channels,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    activation_type=self.activation_type,
                    activation_params=self.activation_params
                )
            else:
                block = ResidualBlock(
                    in_channels=curr_channels,
                    channels=chunk,
                    kernel_sizes=[3] * len(chunk),
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    activation_type=self.activation_type,
                    activation_params=self.activation_params
                )
                
            layers.append(block)
            curr_channels = chunk[-1]
            
            if len(chunk) == P:
                layers.append(POOLINGS[self.pooling_type](**self.pooling_params))
        # ========================
        seq = nn.Sequential(*layers)
        return seq