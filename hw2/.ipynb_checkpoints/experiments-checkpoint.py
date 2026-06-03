import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from utils.train_results import FitResult

from .cnn import CNN, ResNet
from .mlp import MLP
from .training import ClassifierTrainer
from .classifier import ArgMaxClassifier, BinaryClassifier, select_roc_thresh

DATA_DIR = os.path.expanduser("~/datasets")

MODEL_TYPES = {
    ###
    "cnn": CNN,
    "resnet": ResNet,
}


# DEFAULT_EXPERIMENT_CONFIG
# Students may change the values below to try different settings.
# Only edit values; do not rename keys unless you know what you're doing.
from copy import deepcopy

DEFAULT_EXPERIMENT_CONFIG = {
    "out_dir": "./results",
    "bs_train": 128,
    "bs_test": None,
    "batches": 100,
    "epochs": 100,
    "early_stopping": 3,
    "checkpoints": None,
    "lr": 1e-3,
    "reg": 1e-3,
    "pool_every": 2,
    "hidden_dims": (1024,),
    "model_type": "cnn",
    "conv_params": dict(kernel_size=3, padding=1, stride=1),
    "pooling_params": dict(kernel_size=2),
}


def mlp_experiment(
        depth: int,
        width: int,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        dl_test: DataLoader,
        n_epochs: int,
):
    # TODO:
    #  - Create a BinaryClassifier model.
    #  - Train using our ClassifierTrainer for n_epochs, while validating on the
    #    validation set.
    #  - Use the validation set for threshold selection.
    #  - Set optimal threshold and evaluate one epoch on the test set.
    #  - Return the model, the optimal threshold value, the accuracy on the validation
    #    set (from the last epoch) and the accuracy on the test set (from a single
    #    epoch).
    #  Note: use print_every=0, verbose=False, plot=False where relevant to prevent
    #  output from this function.
    # ====== YOUR CODE: ======

    # 1. Create the model
    # The architecture asks for a specific depth and width.
    # We use ReLU activations for hidden layers and no activation ('none') for the output layer
    # since our BinaryClassifier will apply Softmax internally via predict_proba.
    dims = [width] * depth + [2]  # 'depth' hidden layers of size 'width', then output of size 2
    nonlins = ['relu'] * depth + ['none']

    # Get the input dimension from the first batch of the training set
    x0, _ = next(iter(dl_train))
    in_dim = torch.numel(x0[0]) # Get the total number of elements in a single sample
    
    mlp = MLP(in_dim=in_dim, dims=dims, nonlins=nonlins)
    model = BinaryClassifier(model=mlp, threshold=0.5)

    # 2. Train the model
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam usually converges faster here
    trainer = ClassifierTrainer(model, loss_fn, optimizer)

    fit_result = trainer.fit(
        dl_train,
        dl_valid,
        num_epochs=n_epochs,
        print_every=0  # Silent training as requested
    )

    # Extract validation accuracy from the last epoch
    valid_acc = fit_result.test_acc[-1]

    # 3. Threshold selection
    # We use the entire validation dataset to find the optimal threshold
    x_valid, y_valid = dl_valid.dataset.tensors
    thresh = select_roc_thresh(model, x_valid, y_valid, plot=False)

    # Update the model with the optimal threshold
    model.threshold = thresh

    # 4. Evaluate on the test set
    test_result = trainer.test_epoch(dl_test, verbose=False)
    test_acc = test_result.accuracy

    # ========================
    return model, thresh, valid_acc, test_acc


def cnn_experiment(
    run_name,
    out_dir="./results",
    seed=None,
    device=None,
    # Training params
    bs_train=128,
    bs_test=None,
    batches=100,
    epochs=100,
    early_stopping=3,
    checkpoints=None,
    lr=1e-3,
    reg=1e-3,
    # Model params
    filters_per_layer=[64],
    layers_per_block=2,
    pool_every=2,
    hidden_dims=[1024],
    model_type="cnn",
    # You can add extra configuration for your experiments here
    conv_params=dict(kernel_size=3, padding=1, stride=1),  # can be an input
    pooling_params=dict(kernel_size=2) ,  # can be an input
    **kw,
):
    """
    Executes a single run of a Part3 experiment with a single configuration.

    These parameters are populated by the CLI parser below.
    See the help string of each parameter for it's meaning.
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()
    ds_train = CIFAR10(root=DATA_DIR, download=False, train=True, transform=tf)
    ds_test = CIFAR10(root=DATA_DIR, download=False, train=False, transform=tf)

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = MODEL_TYPES[model_type]

    # TODO: Train
    #  - Create model, loss, optimizer and trainer based on the parameters.
    #    Use the model you've implemented previously, cross entropy loss and
    #    any optimizer that you wish.
    #  - Run training and save the FitResults in the fit_res variable.
    #  - The fit results and all the experiment parameters will then be saved
    #   for you automatically.
    fit_res = None
    # ====== YOUR CODE: ======

    # Generate the channels list by repeating the filters_per_layer list L times
    channels = filters_per_layer * layers_per_block

    model = model_cls(
        in_size=ds_train[0][0].shape,
        out_classes=10,
        channels=channels,
        pool_every=pool_every,
        hidden_dims=hidden_dims,
        conv_params=conv_params,
        pooling_params=pooling_params,
        **kw
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    # Adam usually converges better and faster for these deep CNNs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    trainer = ClassifierTrainer(model, loss_fn, optimizer, device)

    dl_train = DataLoader(ds_train, bs_train, shuffle=True)
    dl_test = DataLoader(ds_test, bs_test, shuffle=False)

    fit_res = trainer.fit(
        dl_train, dl_test,
        num_epochs=epochs,
        checkpoints=checkpoints,
        early_stopping=early_stopping,
        max_batches=batches
    )
    # ========================
    return save_experiment(run_name, out_dir, cfg, fit_res)


def part5_exp1(
        seed=None,
        device=None,
):
    # Build config from global defaults
    base = deepcopy(DEFAULT_EXPERIMENT_CONFIG)
    base["seed"] = seed
    base["device"] = device

    configs = []
    # ====== YOUR CODE: ======
    filters_per_layer = [32, 64]
    layers_per_block = [2, 4, 8, 16]

    for k in filters_per_layer:
        for l in layers_per_block:
            cfg = dict(base)
            cfg.update({
                "run_name": f"exp1_L{l}_K{k}",
                "filters_per_layer": [k],
                "layers_per_block": l,
                # Pool every time we finish a full layer block or max every 4 convs to avoid shrinking too fast
                "pool_every": max(1, l // 2) if l <= 8 else 4,
            })
            configs.append(cfg)
    # ========================
    results = []
    for cfg in tqdm(configs, desc="Experiment 1"):
        results.append(cnn_experiment(**cfg))
    return results


def part5_exp2(seed=None, device=None):
    base = deepcopy(DEFAULT_EXPERIMENT_CONFIG)
    base["seed"] = seed
    base["device"] = device

    configs = []
    # ====== YOUR CODE: ======
    filters_per_layer = [[32, 64, 128], [64, 128, 256]]
    layers_per_block = [2, 4, 8]

    for k in filters_per_layer:
        for l in layers_per_block:
            cfg = dict(base)
            cfg.update({
                "run_name": f"exp2_L{l}_K{k[0]}",
                "filters_per_layer": k,
                "layers_per_block": l,
                "pool_every": max(1, l // 2) if l <= 8 else 4,
            })
            configs.append(cfg)
    # ========================
    results = []
    for cfg in tqdm(configs, desc="Experiment 2"):
        results.append(cnn_experiment(**cfg))
    return results


def part5_exp3(seed=None, device=None):
    base = deepcopy(DEFAULT_EXPERIMENT_CONFIG)
    base["seed"] = seed
    base["device"] = device

    configs = []
    # ====== YOUR CODE: ======
    # Testing effect of dropout and batchnorm (which we can pass as **kw)
    filters_per_layer = [64]
    layers_per_block = [2, 4]

    for k in filters_per_layer:
        for l in layers_per_block:
            cfg = dict(base)
            cfg.update({
                "run_name": f"exp3_L{l}_K{k}",
                "filters_per_layer": [k],
                "layers_per_block": l,
                "pool_every": max(1, l // 2),
                "dropout": 0.3,  # Add Dropout
                "batchnorm": True,  # Add BatchNorm
            })
            configs.append(cfg)
    # ========================
    results = []
    for cfg in tqdm(configs, desc="Experiment 3"):
        results.append(cnn_experiment(**cfg))
    return results


def part5_exp4(seed=None, device=None):
    base = deepcopy(DEFAULT_EXPERIMENT_CONFIG)
    base["seed"] = seed
    base["device"] = device
    base["model_type"] = "resnet"  # ResNet for this experiment

    configs = []
    # ====== YOUR CODE: ======
    # ResNet allows us to train very deep networks without vanishing gradients
    filters_per_layer = [32]
    layers_per_block = [8, 16, 32]  # Going very deep!

    for k in filters_per_layer:
        for l in layers_per_block:
            cfg = dict(base)
            cfg.update({
                "run_name": f"exp4_resnet_L{l}_K{k}",
                "filters_per_layer": [k],
                "layers_per_block": l,
                "pool_every": 4,  # Pool every 4 layers so we don't shrink to 0
                "batchnorm": True,
            })
            configs.append(cfg)
    # ========================
    results = []
    for cfg in tqdm(configs, desc="Experiment 4"):
        results.append(cnn_experiment(**cfg))
    return results


def save_experiment(run_name, out_dir, cfg, fit_res):
    cfg['device'] = str(cfg['device'])
    output = dict(config=cfg, results=fit_res._asdict())

    cfg_LK = (
        f'L{cfg["layers_per_block"]}_K'
        f'{"-".join(map(str, cfg["filters_per_layer"]))}'
    )
    output_filename = f"{os.path.join(out_dir, run_name)}_{cfg_LK}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"*** Output file {output_filename} written")
    return output_filename


def load_experiment(filename):
    with open(filename, "r") as f:
        output = json.load(f)

    config = output["config"]
    fit_res = FitResult(**output["results"])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description="HW2 Experiments")
    sp = p.add_subparsers(help="Sub-commands")

    # Experiment config
    sp_exp = sp.add_parser(
        "run-exp", help="Run experiment with a single " "configuration"
    )
    sp_exp.set_defaults(subcmd_fn=cnn_experiment)
    sp_exp.add_argument(
        "--run-name", "-n", type=str, help="Name of run and output file", required=True
    )
    sp_exp.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        default="./results",
        required=False,
    )
    sp_exp.add_argument(
        "--seed", "-s", type=int, help="Random seed", default=None, required=False
    )
    sp_exp.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device (default is autodetect)",
        default=None,
        required=False,
    )

    # # Training
    sp_exp.add_argument(
        "--bs-train",
        type=int,
        help="Train batch size",
        default=128,
        metavar="BATCH_SIZE",
    )
    sp_exp.add_argument(
        "--bs-test", type=int, help="Test batch size", metavar="BATCH_SIZE"
    )
    sp_exp.add_argument(
        "--batches", type=int, help="Number of batches per epoch", default=100
    )
    sp_exp.add_argument(
        "--epochs", type=int, help="Maximal number of epochs", default=100
    )
    sp_exp.add_argument(
        "--early-stopping",
        type=int,
        help="Stop after this many epochs without " "improvement",
        default=3,
    )
    sp_exp.add_argument(
        "--checkpoints",
        type=int,
        help="Save model checkpoints to this file when test " "accuracy improves",
        default=None,
    )
    sp_exp.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    sp_exp.add_argument("--reg", type=float, help="L2 regularization", default=1e-3)

    # # Model
    sp_exp.add_argument(
        "--filters-per-layer",
        "-K",
        type=int,
        nargs="+",
        help="Number of filters per conv layer in a block",
        metavar="K",
        required=True,
    )
    sp_exp.add_argument(
        "--layers-per-block",
        "-L",
        type=int,
        metavar="L",
        help="Number of layers in each block",
        required=True,
    )
    sp_exp.add_argument(
        "--pool-every",
        "-P",
        type=int,
        metavar="P",
        help="Pool after this number of conv layers",
        required=True,
    )
    sp_exp.add_argument(
        "--hidden-dims",
        "-H",
        type=int,
        nargs="+",
        help="Output size of hidden linear layers",
        metavar="H",
        required=True,
    )
    sp_exp.add_argument(
        "--model-type",
        "-M",
        choices=MODEL_TYPES.keys(),
        default="cnn",
        help="Which model instance to create",
    )

    parsed = p.parse_args()

    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == "__main__":
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f"*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}")
    subcmd_fn(**vars(parsed_args))
