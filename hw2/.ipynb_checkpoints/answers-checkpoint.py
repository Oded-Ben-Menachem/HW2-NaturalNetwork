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
    wstd, lr, reg = 0.1, 0.1, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.09,
        0.05,
        0.0005,
        0.001,
        0.0005,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
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
        0.01,
        0.005,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
Yes, the results match our expectations.

1. With dropout=0, the model heavily overfits. The train accuracy climbs to 67.8% and train loss drops below 1.0, but the test loss jumps to 2.91 with a low test accuracy of 22.2%. The model memorized the small dataset instead of generalizing.
With dropout=0.4, dropout improves generalization. It makes training harder, so train accuracy is lower (30.0%), but it achieves a better and more stable test loss (2.08) and a higher test accuracy (23.9%).

2. dropout=0.4 (Low): Finds the right balance. It prevents overfitting while maintaining enough network capacity to learn useful features.
dropout=0.8 (High): Causes severe underfitting. Turning off 80% of the neurons completely kills the model's capacity, leaving both train and test accuracy at a random guess level (10%-15%).



"""

part2_q2 = r"""
Yes it is passible.
The reason is that Loss measures confidence, while Accuracy only measures correctness.
Accuracy is a step-function. It only cares if the correct class got the highest score, completely ignoring the exact probability values.
Cross-Entropy Loss is a continuous function based on the exact probabilities. It heavily penalizes low-confidence predictions via the log function
"""

part2_q3 = r"""
1. Back-propagation is the algorithm used to calculate the gradients of the loss function with respect to the model's weights. It is a smart application of the calculus Chain Rule, moving backward from the output layer to the input layer.
Gradient Descent is the optimization algorithm that uses those calculated gradients to actually update the model's weights  in order to minimize the loss.

2. Gradient Descent Computes the loss and gradients using the entire dataset before making a single weight update.
Stochastic Gradient Descent computes the loss and gradients for a single random sample (or a small mini-batch) and updates the weights immediately.

3. a. Deep learning datasets are huge and cannot fit into RAM/GPU memory all at once. SGD only requires loading a small batch at a time.
b. GD takes a long time just to make one single update. SGD updates the weights frequently, leading to much faster convergence in practice.
c. Pure GD is deterministic and can get stuck in sharp local minima or saddle points. The randomness of mini-batch SGD introduces "noise" into the loss surface, which helps the optimizer bounce out of bad local regions.

4. a. Yes. Because finding the gradient is a linear operation, the gradient of a sum is equal to the sum of the gradients. Summing the losses over disjoint batches and doing one backward pass at the end yields the exact same total gradient vector as standard Batch GD.
b. Even though each batch processes the data, the calculation at the end of each batch is not cleared from the memory. Instead, it waits for all the samples to finish. Because of this, a lot of data is saved in the memory at the same time, and this causes the memory to crash.

"""

part2_q4 = r"""
1. a. Yes. Because we don't need to save the memory of the previous layers and we can calculate the derivative immediately at each step, the memory complexity for this method is  O(1). This way, we can do the calculation at each layer and simply overwrite the memory.
b. To save memory, we can save the value every few layers (every $\sqrt{n}$ layers), and then during the backward pass, we recompute the missing layers from these checkpoints. The time complexity is still $O(n)$, but the space (memory) complexity becomes $O(\sqrt{n})$.

2. Forward Mode: No, it is more complicated for arbitrary graphs because there can be multiple splits to different layers. This makes it problematic to delete data that might still be needed later by another path.
Backward Mode: Yes, we can still choose key points (checkpoints) in the graph and use them to recompute the missing parts.

3. Deep networks like VGG or ResNet use a huge amount of GPU memory because they save the activations for every layer. By using checkpointing, we can save a lot of memory. This allows us to train much deeper models or use larger batch sizes on the same GPU, with only a small cost in training time.
"""

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
