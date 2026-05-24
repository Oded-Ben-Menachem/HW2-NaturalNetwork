r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""

1.
A. If we have a batch of 64, the input is 1024 and the output is 512, then $X$ will be of size $64 \times 1024$, $W$ will be of size $512 \times 1024$, and $Y$ will be of size $64 \times 512$. The dimension of the tensor is the dimension of $X$ times the dimension of $Y$, meaning a 4D tensor of dimension $64 \times 1024 \times 64 \times 512$.

B. Yes. Most elements will be 0 because the batches are not related, and derivatives between different batches will be 0.

C. As we saw in the assignment, there is no need to store the tensor in memory. Instead, using the chain rule and taking the parts that are not 0 in a matrix multiplication, we can generate the derivatives that we need and do not zero out.

2. 

A.If we have a batch of 64, the input is 1024 and the output is 512, then $X$ will be of size $64 \times 1024$, $W$ will be of size $512 \times 1024$, and $Y$ will be of size $64 \times 512$. The dimension of the tensor is the dimension of $Y$ times the dimension of $W$, meaning a 4D tensor of dimension $64 \times 512 \times 512 \times 1024$.

B. Yes. Most elements will be 0 because the weights of a specific neuron only affect the output of that specific neuron, and derivatives between different neurons will be 0.

C. As we saw in the assignment, there is no need to store the tensor in memory. Instead, using the chain rule and taking the parts that are not 0 in a matrix multiplication, we can generate the derivatives that we need and do not zero out.
."""

part1_q2 = r"""
No, it is not required to use backpropagation to train a model. The derivatives are required, but there are other methods to get the derivatives in different variations that can be used to train a model.
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""

Fill me. You can use Markdown and LaTeX in this string."""

part2_q2 = r"""
Fill me. You can use Markdown and LaTeX in this string."""

part2_q3 = r"""
Fill me. You can use Markdown and LaTeX in this string."""

part2_q4 = r"""
Fill me. You can use Markdown and LaTeX in this string."""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
Fill me. You can use Markdown and LaTeX in this string."""

part3_q2 = r"""
Fill me. You can use Markdown and LaTeX in this string.
"""

part3_q3 = r"""
Fill me. You can use Markdown and LaTeX in this string."""


part3_q4 = r"""
Fill me. You can use Markdown and LaTeX in this string."""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
Fill me. You can use Markdown and LaTeX in this string."""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
Fill me. You can use Markdown and LaTeX in this string."""

part5_q2 = r"""
Fill me. You can use Markdown and LaTeX in this string."""

part5_q3 = r"""
Fill me. You can use Markdown and LaTeX in this string."""

part5_q4 = r"""
Fill me. You can use Markdown and LaTeX in this string."""


# ==============
