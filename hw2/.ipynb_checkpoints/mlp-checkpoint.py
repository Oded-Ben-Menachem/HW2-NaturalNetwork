import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}

# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)


class MLP(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(
            self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]]
    ):
        """
        :param in_dim: Input dimension.
        :param dims: Hidden dimensions, including output dimension.
        :param nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
        """
        super().__init__()  # Don't forget to call the base class init!
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]

        # TODO:
        #  - Initialize the layers according to the requested dimensions. Use
        #    either nn.Linear layers or create W, b tensors per layer and wrap them
        #    with nn.Parameter.
        #  - Either instantiate the activations based on their name or use the provided
        #    instances.
        # ====== YOUR CODE: ======
        layers = []
        current_in_dim = in_dim

        for dim, nonlin in zip(dims, nonlins):
            # 1. Add the linear layer
            layers.append(nn.Linear(current_in_dim, dim))

            # 2. Add the activation
            if isinstance(nonlin, str):
                activation_name = nonlin.lower()
                activation_cls = ACTIVATIONS[activation_name]
                kwargs = ACTIVATION_DEFAULT_KWARGS[activation_name]
                layers.append(activation_cls(**kwargs))
            elif isinstance(nonlin, nn.Module):
                layers.append(nonlin)

            # Update the input dimension for the next layer
            current_in_dim = dim

        # Wrap everything in nn.Sequential and assign to a module attribute
        self.model = nn.Sequential(*layers)
        # ========================

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        # TODO: Implement the model's forward pass. Make sure the input and output
        #  shapes are as expected.
        # ====== YOUR CODE: ======
        # Reshape just in case x is passed as a multi-dimensional tensor (e.g. images)
        x = x.view(x.size(0), -1)
        out = self.model(x)
        return out
        # ========================