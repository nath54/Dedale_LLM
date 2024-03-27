
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib import device, tokenizer, config, printd
from lib_data import DataContainer

from random import choice as rchoice

from model import MixtofExp


# debug function, just here if I need it later
def debug_show_trainable_params(model: nn.Module):
    for n, p in model.named_parameters():
        print(n, p.shape, p.requires_grad)


# Function for freezing or unfreezing params from learning
def set_grad_params(module: nn.Module, value: bool):
    for param in module.parameters():
        param.requires_grad = value


# Training function for N epochs on a particular dataset
def training_simple_epochs(
    model: MixtofExp,
    training_config: dict,
    data_container: DataContainer
):
    #
    assert "nb_epochs" in training_config
    assert "datasets_used" in training_config
    assert len(training_config["datasets_used"]) > 0

    # Default values
    if "batch_size" not in training_config:
        training_config["batch_size"] = 1

    if "learning_rate" not in training_config:
        training_config["learning_blocks"] = 0.01

    if "freeze_blocks" not in training_config:
        training_config["freeze_blocks"] = []

    if "freeze_embeddings" not in training_config:
        training_config["freeze_embeddings"] = 0

    if "freeze_next_token_prediction" not in training_config:
        training_config["freeze_next_token_prediction"] = 0

    if "freeze_router" not in training_config:
        training_config["freeze_router"] = 0

    if "forwards_per_epoch" not in training_config:
        training_config["forwards_per_epoch"] = 100

    if "force_passage" not in training_config:
        if "force_passage" in config:
            training_config["force_passage"] = config["force_passage"]
        else:
            training_config["force_passage"] = []

    # Apply configs on model
    set_grad_params(
        model.next_token_prediction,
        True if training_config["freeze_next_token_prediction"] else False
    )
    set_grad_params(
        model.embedding,
        True if training_config["freeze_embeddings"] else False
    )
    set_grad_params(
        model.routeur,
        True if training_config["freeze_router"] else False
    )

    #
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"]
    )
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    tb: SummaryWriter = SummaryWriter()
    
    #
    for epoch in range(training_config["nb_epochs"]):

        losses_epoch = []

        print("epoch : ", epoch)

        loop = tqdm(
            range(training_config["forwards_per_epoch"]),
            leave=False,
            total=training_config["forwards_per_epoch"]
        )

        for i in loop:

            X, Y = data_container.get_data(
                nb_batch=training_config["batch_size"],
                key = rchoice(training_config["datasets_used"])
            )
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()

            output = model(X).to(device)

            loss = loss_fn(output, Y)
            loss.backward()
            losses_epoch.append(loss.item())

            optimizer.step()

            # Show progress while training
            loop.set_description(
                f'Epoch={epoch}/{training_config["nb_epochs"]}'
            )
            loop.set_postfix(loss=loss.item())

        tb.add_scalar("Loss", sum(losses_epoch)/len(losses_epoch), epoch)

        model.save_weights()


# Main train function
def train(model: MixtofExp):
    #
    assert isinstance(model, MixtofExp)
    #
    training_configs: list[dict] = config["training"]
    #
    print("Preparing the model to train...")
    model.to(device)
    model.train()

    #
    dc: DataContainer = DataContainer(
        config["data"],
        size_streaming=1000,
        prompt_per_stream=100,
        randomized_access=True,
        tokenizer=tokenizer,
        padding_context_length=config["context_length"]
    )

    #
    for t in training_configs:
        #
        if "strategy" not in t:
            print("Warning: no 'strategy' field in the config, ignored.")
            continue
        #
        if t["strategy"] == "fixed_nb_of_epochs":
            training_simple_epochs(model, t, dc)
