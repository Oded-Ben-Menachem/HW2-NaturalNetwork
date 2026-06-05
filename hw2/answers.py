r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""

1.
A. If we have a batch of 64, the input is 1024 and the output is 512, then $X$ will be of size $64 \times 1024$, $W$ will be of size $512 \times 1024$, and $Y$ will be of size $64 \times 512$. The dimension of the tensor is the dimension of Y times the dimension of X, meaning a 4D tensor of dimension 64 x 512 x 64 x 1024.

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
    n_layers = 2  # number of hidden layers
    hidden_dims = 64  # number of output dimensions for each hidden layer
    activation = "relu"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer (CrossEntropy needs raw scores)

    # ====== YOUR CODE: ======
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

    loss_fn = torch.nn.CrossEntropyLoss()  # Standard loss for classification
    
    # שינינו את קצב הלמידה והגברנו רגולריזציה כדי למנוע אוברפיטינג
    lr, weight_decay, momentum = 0.005, 0.01, 0.9  # Arguments for SGD optimizer

    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)

part3_q1 = r"""
Based on the plots (loss/accuracy curves and decision boundary), the qualitative assessment of the errors is as follows:

1. **Optimization Error is LOW:** The training loss decreases smoothly and converges to a low value, while training accuracy reaches a high level (e.g., >85%). This indicates that our optimization algorithm (Adam/SGD) successfully minimized the loss function and found a good local minimum without getting stuck.

2. **Generalization Error is LOW:** The gap between the training accuracy/loss and the validation accuracy/loss is relatively small. The validation metrics follow the training metrics closely, indicating that the model did not memorize the training data (no severe overfitting) and generalizes well to unseen data from the same mixture distribution.

3. **Approximation Error is LOW:** The plotted decision boundary is highly non-linear and successfully captures the complex, overlapping "crescent" shapes of the moon dataset. This shows that the MLP architecture we chose (its depth, width, and ReLU activations) has sufficient capacity to approximate the underlying true function of the data.
"""

part3_q2 = r"""
Based on the data generating process, we would expect an imbalance between the False Positive Rate (FPR) and False Negative Rate (FNR). 

The validation set is drawn from a mixture of two `make_moons` distributions (rotated 10 degrees with 0.2 noise, and rotated 50 degrees with 0.25 noise). Because the second distribution has a very strong rotation and higher noise, one of the "moons" (either class 0 or class 1) is pushed deeper into the spatial territory of the other class. 
Since the standard threshold is rigidly set at 0.5, the model's decision boundary will try to smoothly separate the classes. The class that is more "smeared" or spread out into the other's region due to the 50-degree rotation will suffer from more misclassifications. If Class 0 (negative) points are pushed into the Class 1 region, the **FPR will be higher**. Conversely, if Class 1 (positive) points are pushed into the Class 0 region, the **FNR will be higher**.
"""

part3_q3 = r"""
In real-world scenarios, the threshold is dictated by the specific cost matrix of False Positives (FP) vs. False Negatives (FN). We would **not** use the same "optimal" point (Youden's J statistic) from the generic ROC analysis.

1. **Scenario 1 (Non-lethal, expensive further testing):** Here, the cost of a False Positive is very high (sending a healthy person to an expensive, high-risk test), while the cost of a False Negative is relatively low (the disease is non-lethal, will show symptoms later, and is treatable). 
**Adjustment:** We would choose a **higher threshold**. On the ROC curve, we would move to the bottom-left. This drastically reduces the FPR (protecting healthy patients from dangerous tests) at the acceptable cost of a lower True Positive Rate (TPR).

2. **Scenario 2 (Silent, lethal disease):** Here, the cost of a False Negative is astronomical (the patient dies), while the cost of a False Positive is acceptable compared to loss of life. We want to catch every single sick person.
**Adjustment:** We would choose a **lower threshold**. On the ROC curve, we would move to the top-right. This maximizes TPR (Sensitivity) close to 1.0, ensuring almost no sick patients are missed, even though it results in a high FPR (many healthy people will undergo unnecessary tests).
"""


part3_q4 = r"""
1. **Fixed Depth, Varying Width (Columns):** As the width (number of neurons per layer) increases, the model's capacity grows. A very narrow network (e.g., width=2) severely underfits, resulting in a nearly linear and inaccurate decision boundary. As width increases to 8 and 32, the model can capture more localized, complex features, resulting in a decision boundary that tightly fits the curved shape of the moons and higher accuracy.

2. **Fixed Width, Varying Depth (Rows):** Increasing the depth (number of layers) allows the network to learn hierarchical and more abstract compositional representations. A shallow network (depth=1) acts like a basic combination of linear hyperplanes. As depth increases to 4, the decision boundary becomes highly non-linear, creating complex "folds" and islands that isolate data points much better.

3. **Depth=1, Width=32 vs. Depth=4, Width=8:** Both networks have roughly a similar number of parameters. However, the deep/narrow network (depth=4, width=8) often creates highly non-linear and compositional boundaries, which is generally better for complex topological shapes like intertwined moons. The shallow/wide network (depth=1, width=32) acts more like a template matcher; it produces smoother boundaries but struggles more to capture highly abstract representations compared to the deep network.

4. **Effect of Threshold Selection:** Yes, optimizing the threshold on the validation set generally **improves** the results on the test set. The default 0.5 threshold assumes the model is perfectly calibrated and the classes are perfectly balanced. However, due to noise and rotation, the model's probabilities are often slightly biased. Using ROC analysis on the validation set finds the true density separation point. Because the test set is related to the validation distribution, applying this calibrated threshold corrects the model's inherent bias and increases test accuracy.
"""

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
    loss_fn = torch.nn.CrossEntropyLoss()
    lr, weight_decay, momentum = 0.1, 0.0, 0.0
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)

part4_q1 = r"""
1. **Number of Parameters:**
   - **Regular Block:** Uses two $3\times3$ convolutions, each mapping 256 input channels to 256 output channels. 
     Parameters = $2 \times (3 \times 3 \times 256 \times 256) = 2 \times 589,824 = \mathbf{1,179,648}$ (ignoring biases).
   - **Bottleneck Block:** Uses three convolutions: 
     a $1\times1$ projection (256 $\rightarrow$ 64), a $3\times3$ inner conv (64 $\rightarrow$ 64), and a $1\times1$ projection (64 $\rightarrow$ 256).
     Parameters = $(1 \times 1 \times 256 \times 64) + (3 \times 3 \times 64 \times 64) + (1 \times 1 \times 64 \times 256) = 16,384 + 36,864 + 16,384 = \mathbf{69,632}$ (ignoring biases).
     *The bottleneck block uses about 17x fewer parameters.*

2. **Floating Point Operations (FLOPS):**
   - The number of FLOPS is directly proportional to the number of parameters multiplied by the spatial dimensions of the feature map. Since the bottleneck block drastically reduces the number of parameters by operating its heavy $3\times3$ convolution on a much smaller channel depth (64 instead of 256), its computational cost (FLOPS) is significantly lower compared to the regular block.

3. **Ability to combine the input:**
   - **Spatially (within feature maps):** Both blocks combine spatial information using the $3\times3$ convolution. However, the regular block does this twice (larger effective receptive field), while the bottleneck block only has one $3\times3$ spatial convolution.
   - **Across feature maps (cross-channel):** The bottleneck block excels here. The $1\times1$ convolutions are explicitly designed to act as fully connected layers applied to each pixel, forcing the network to combine and compress cross-channel features efficiently before and after the spatial operation. The regular block combines them naturally, but without the explicit dimension reduction/expansion phase.
"""
# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
1. Increasing depth allows the network to learn more complex features, but makes it harder to train. The depth that produced the best results is l=2 , as it provides the best trade off. deep enough to extract meaningful features, but shallow enough to maintain a healthy gradient flow.

2. For deep configurations L=8 and L=16, the network became untrainable, and accuracy stalled at 10\%. This is caused by the Vanishing Gradient problem. During backpropagation, the gradients exponentially decrease as they pass through many layers, leaving the earliest layers with zero updates and unable to learn.
Two possible solutions are to add normalization layers or to use residual connections.
"""

part5_q2 = r"""
3. Yes, L=4 became trainable and L=2 showed significant improvement compared to the baseline in experiment 1. Both networks converged faster and reached higher accuracies (around 70% for L=2).
The reason for this improvement is the addition of Batch Normalization. Batch Normalization addresses the problem of internal covariate shift by normalizing the inputs of each layer, ensuring a stable distribution. This prevents gradients from vanishing or exploding early in the training process, allowing deeper networks like L=4 to learn effectively. Additionally, the inclusion of Dropout acted as a regularizer, preventing overfitting and improving generalization to the test set.

4. The ResNet architecture (experiment 4) demonstrated a massive improvement in training very deep networks compared to both experiment 1 and 3. In experiment 1, networks deeper than L=4 completely collapsed (stuck at 10% accuracy due to vanishing gradients). In experiment 3, we managed to stabilize L=4 using BatchNorm. 
However, ResNet allowed us to successfully train extremely deep networks (e.g., L=16 and L=32) with a steady decrease in loss and high accuracy. This is because the skip connections (shortcuts) in the residual blocks create an "express lane" for gradients during backpropagation. This allows gradients to flow directly to earlier layers without diminishing, effectively solving the vanishing gradient problem structurally and enabling the benefits of deep feature extraction without the training penalties.
"""

part5_q3 = r"""
5. To improve the ResNet's performance, we could implement several strategies:
- Data Augmentation: Applying transformations like random crops, flips, and rotations to the training set to prevent overfitting and improve generalization.
- Learning Rate Scheduling: Starting with a higher learning rate and decaying it over time (e.g., using a StepLR or Cosine Annealing scheduler) can help the model converge to a better local minimum.
- Advanced Architectures: Instead of basic residual blocks, we could use bottleneck blocks (which reduce dimensionality and computation) as seen in deeper ResNets (like ResNet-50).
- Regularization: Increasing Dropout rates slightly or fine-tuning weight decay (L2 regularization) can further reduce overfitting on smaller datasets like CIFAR-10.
"""

part5_q4 = r"""
6. If we had a larger computational budget and larger datasets (like ImageNet), the fundamental concepts would remain exactly the same. The principles of convolutions, pooling, residual connections, and backpropagation scale perfectly. 
However, the scale of the implementation would change significantly:
- Deeper and Wider Networks: We could train models with hundreds of layers (e.g., ResNet-101 or ResNet-152) and more channels, allowing the network to learn vastly more complex and hierarchical features.
- Larger Batch Sizes: With more VRAM (GPUs), we could use larger batch sizes, which stabilizes the gradient updates and speeds up training.
- Longer Training: Models would be trained for hundreds of epochs over days or weeks, requiring more advanced learning rate scheduling and early stopping mechanisms.
- While the core math and architecture stay the same, the engineering required to handle the data pipeline and distributed training across multiple GPUs becomes the primary challenge.
"""


# ==============
